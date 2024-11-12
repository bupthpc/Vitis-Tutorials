//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#include <adf.h>
#include <aie_api/aie.hpp>

#include "conv2d_w5B.h"


// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

conv2d_w5B::conv2d_w5B( void )
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
  state = 0;
  auto itw = aie::begin_vector<16>(data);
  for (unsigned ii=0; ii < 768/16; ii++)
    chess_prepare_for_pipelining
  {
    *itw++ = aie::zeros<bfloat16,16>();
  }
}

// ------------------------------------------------------------
// Grab Weights
// ------------------------------------------------------------

void conv2d_w5B::grab_wts( input_async_buffer<bfloat16>& wts_i )
{
  if (state == 0) {
    wts_i.acquire();
    state = 1;
  }
}

// ------------------------------------------------------------
// Filter 3x3
// ------------------------------------------------------------

inline __attribute__((noinline))
void conv2d_w5B::filter_3x3( input_buffer<bfloat16>& ifm_i, input_async_buffer<bfloat16>& wts_i )
{
  std::array<aie::vector<bfloat16,32>,4> buffc;
  std::array<aie::vector<bfloat16,32>,3> buffw;
  std::array<aie::accum<accfloat,16>,6>  acc;
  aie::vector<bfloat16,32>               zerobuff = aie::zeros<bfloat16,32>();
  aie::vector<bfloat16,32>               bias;

  auto idat = aie::begin_restrict_vector<16>(ifm_i);
  auto itw  = aie::begin_vector<16>(data);

  // Loop over 3 output image rows:
  for (unsigned rr=0; rr < 3; rr++)
    chess_prepare_for_pipelining
  {
    // Reset the weight pointer:
    auto itap = aie::begin_restrict_vector<16>(wts_i);

    // Loop over 32 of 128 output channels computing 4 at a time:
    for (unsigned oc=0; oc < 32; oc+=4)
      chess_prepare_for_pipelining
    {
      // Set bias weight for these output layers:
      // --> load 4 bias x 8 copies, shuffle takes one from each bias repeatedly, lower 16 matches 'acc' format:
      bias = aie::concat(aie::broadcast<bfloat16,8>(*(wts_i.data()+18432+oc)),
                         aie::broadcast<bfloat16,8>(*(wts_i.data()+18432+oc+1)),
                         aie::broadcast<bfloat16,8>(*(wts_i.data()+18432+oc+2)),
                         aie::broadcast<bfloat16,8>(*(wts_i.data()+18432+oc+3)));
      bias = shuffle(bias,shuffle_T16_4x8);
      acc[0] = aie::zeros<accfloat,16>();
      acc[1] = aie::zeros<accfloat,16>();
      acc[2] = aie::zeros<accfloat,16>();
      acc[3] = aie::zeros<accfloat,16>();
      acc[4] = aie::zeros<accfloat,16>();
      acc[5] = aie::zeros<accfloat,16>();

      // Loop over 64 input channels computing 8 at a time:
      for (unsigned ic=0,off=0; ic < 64; ic += 8)
        chess_prepare_for_pipelining
      {
        // Iteration 0 --> w0, w1, w2
        // Iteration 1 --> w3, w4, w5
        // Iteration 2 --> w6, w7, w8
        for (unsigned tt=0,idx=off; tt < 3; tt++)
          chess_prepare_for_pipelining
        {
          buffc[0].insert(0,*(idat+idx  ));                      buffw[0].insert(0,*(itap++));
          buffc[0].insert(1,*(idat+idx+1));                      buffw[0].insert(1,*(itap++));
          buffc[1].insert(0,*(idat+idx+2));                      buffw[1].insert(0,*(itap++));
          buffc[1].insert(1,*(idat+idx+3));                      buffw[1].insert(1,*(itap++));
          buffw[2].insert(0,*(itap++));
          buffw[2].insert(1,*(itap++));
          acc[0] = mac_4x8_8x4(                       buffc[0],             buffw[0],acc[0]);
          acc[1] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[0],buffc[1], 8),buffw[1],acc[1]);
          acc[2] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[0],buffc[1],16),buffw[2],acc[2]);
          acc[3] = mac_4x8_8x4(                       buffc[1],             buffw[0],acc[3]);
          acc[4] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[1],zerobuff, 8),buffw[1],acc[4]);
          acc[5] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[1],zerobuff,16),buffw[2],acc[5]);
          idx += 32; // Advance down by 1 input image row
        } // tt
        off += 4; // Advance to next 8 'ic' input layers
      } // ic
      acc[0] = aie::add(aie::add(acc[0],acc[1]),aie::add(acc[2],aie::accum<accfloat,16>(bias.extract<16>(0))));
      acc[3] = aie::add(aie::add(acc[3],acc[4]),aie::add(acc[5],aie::accum<accfloat,16>(bias.extract<16>(0))));
      *itw++ = aie::max(acc[0].to_vector<bfloat16>(),bfloat16(0));
      *itw++ = aie::max(acc[3].to_vector<bfloat16>(),bfloat16(0));
    } // oc
    idat += 32; // Advance by 1 input image row
  } // rr
}

// ------------------------------------------------------------
// Pass Outputs
// ------------------------------------------------------------

inline __attribute__((noinline))
void conv2d_w5B::pass_outputs( input_stream<bfloat16>* pass_i, output_stream<bfloat16>* pass_o )
{
  auto itr = aie::begin_vector<16>(data);
  for (unsigned ii=0; ii < 768/16; ii++)
    chess_prepare_for_pipelining
  {
    writeincr(pass_o,readincr_v<16>(pass_i));
  }
  for (unsigned ii=0; ii < 768/16; ii++)
    chess_prepare_for_pipelining
  {
    writeincr(pass_o,*itr++);
  }
}

// ------------------------------------------------------------
// Run
// ------------------------------------------------------------

void conv2d_w5B::run( input_buffer<bfloat16>& ifm_i,
                      input_async_buffer<bfloat16>& wts_i,
                      input_stream<bfloat16>* pass_i,
                      output_stream<bfloat16>* pass_o )
{
  grab_wts( wts_i );
  filter_3x3( ifm_i, wts_i );
  pass_outputs( pass_i, pass_o );
}


