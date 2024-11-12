//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#include <adf.h>
#include <aie_api/aie.hpp>

#include "conv2d_w3.h"


// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

conv2d_w3::conv2d_w3( void )
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
  fsm_state = 0;
}

// ------------------------------------------------------------
// Grab Weights
// ------------------------------------------------------------

void conv2d_w3::grab_wts( input_async_buffer<bfloat16>& wts_i )
{
  if (fsm_state == 0) {
    wts_i.acquire();
    fsm_state = 1;
  }
}

// ------------------------------------------------------------
// Filter 3x3
// ------------------------------------------------------------

void conv2d_w3::filter_3x3( input_buffer<bfloat16>& ifm_i,
                            input_async_buffer<bfloat16>& wts_i,
                            output_buffer<bfloat16>& ofm_o )
{
  std::array<aie::vector<bfloat16,32>,4> buffc;
  std::array<aie::vector<bfloat16,32>,3> buffw;
  std::array<aie::accum<accfloat,16>,4>  acc;
  aie::vector<bfloat16,32>               zerobuff = aie::zeros<bfloat16,32>();
  aie::vector<bfloat16,32>               bias;

  auto idat = aie::begin_restrict_vector<16>(ifm_i);
  auto itw  = aie::begin_vector<16>(ofm_o);

  // Loop over 11 output image rows:
  for (unsigned rr=0; rr < 11; rr++)
    chess_prepare_for_pipelining
  {
    // Reset the weight pointer:
    auto itap = aie::begin_restrict_vector<16>(wts_i);

    // Loop over 64 output channels computing 4 at a time:
    for (unsigned oc=0; oc < 64; oc+=4)
      chess_prepare_for_pipelining
    {
      // Set bias weight for these output layers:
      // --> load 4 bias x 8 copies, shuffle takes one from each bias repeatedly, lower 16 matches 'acc' format:
      bias = aie::concat(aie::broadcast<bfloat16,8>(*(wts_i.data()+9216+oc)),
                         aie::broadcast<bfloat16,8>(*(wts_i.data()+9216+oc+1)),
                         aie::broadcast<bfloat16,8>(*(wts_i.data()+9216+oc+2)),
                         aie::broadcast<bfloat16,8>(*(wts_i.data()+9216+oc+3)));
      bias = shuffle(bias,shuffle_T16_4x8);
      acc[0].from_vector(bias.extract<16>(0));
      acc[1].from_vector(bias.extract<16>(0));
      acc[2].from_vector(bias.extract<16>(0));
      acc[3].from_vector(bias.extract<16>(0));

      // -------------------- First 8 input channels --------------------

      // Iteration 0 --> w0, w1, w2
      // Iteration 1 --> w3, w4, w5
      // Iteration 2 --> w6, w7, w8
      for (unsigned tt=0; tt < 3*16; tt+=16)
        chess_prepare_for_pipelining
      {
        buffc[0].insert(0,*(idat+tt  ));                      buffw[0].insert(0,*(itap++));
        buffc[0].insert(1,*(idat+tt+1));                      buffw[0].insert(1,*(itap++));
        buffc[1].insert(0,*(idat+tt+2));                      buffw[1].insert(0,*(itap++));
        buffc[1].insert(1,*(idat+tt+3));                      buffw[1].insert(1,*(itap++));
        buffc[2].insert(0,*(idat+tt+4));                      buffw[2].insert(0,*(itap++));
        buffc[2].insert(1,*(idat+tt+5));                      buffw[2].insert(1,*(itap++));
        buffc[3].insert(0,*(idat+tt+6));
        buffc[3].insert(1,*(idat+tt+7));
        acc[0] = mac_4x8_8x4(buffc[0],buffw[0],acc[0]);
        acc[1] = mac_4x8_8x4(buffc[1],buffw[0],acc[1]);
        acc[2] = mac_4x8_8x4(buffc[2],buffw[0],acc[2]);
        acc[3] = mac_4x8_8x4(buffc[3],buffw[0],acc[3]);
        acc[0] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[0],buffc[1], 8),buffw[1],acc[0]);
        acc[1] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[1],buffc[2], 8),buffw[1],acc[1]);
        acc[2] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[2],buffc[3], 8),buffw[1],acc[2]);
        acc[3] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[3],zerobuff, 8),buffw[1],acc[3]);
        acc[0] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[0],buffc[1],16),buffw[2],acc[0]);
        acc[1] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[1],buffc[2],16),buffw[2],acc[1]);
        acc[2] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[2],buffc[3],16),buffw[2],acc[2]);
        acc[3] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[3],zerobuff,16),buffw[2],acc[3]);
      }

      // -------------------- Second 8 input channels --------------------
      // Iteration 0 --> w0, w1, w2
      // Iteration 1 --> w3, w4, w5
      // Iteration 2 --> w6, w7, w8
      for (unsigned tt=8; tt < 3*16+8; tt+=16)
        chess_prepare_for_pipelining
      {
        buffc[0].insert(0,*(idat+tt  ));                      buffw[0].insert(0,*(itap++));
        buffc[0].insert(1,*(idat+tt+1));                      buffw[0].insert(1,*(itap++));
        buffc[1].insert(0,*(idat+tt+2));                      buffw[1].insert(0,*(itap++));
        buffc[1].insert(1,*(idat+tt+3));                      buffw[1].insert(1,*(itap++));
        buffc[2].insert(0,*(idat+tt+4));                      buffw[2].insert(0,*(itap++));
        buffc[2].insert(1,*(idat+tt+5));                      buffw[2].insert(1,*(itap++));
        buffc[3].insert(0,*(idat+tt+6));
        buffc[3].insert(1,*(idat+tt+7));
        acc[0] = mac_4x8_8x4(buffc[0],buffw[0],acc[0]);
        acc[1] = mac_4x8_8x4(buffc[1],buffw[0],acc[1]);
        acc[2] = mac_4x8_8x4(buffc[2],buffw[0],acc[2]);
        acc[3] = mac_4x8_8x4(buffc[3],buffw[0],acc[3]);
        acc[0] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[0],buffc[1], 8),buffw[1],acc[0]);
        acc[1] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[1],buffc[2], 8),buffw[1],acc[1]);
        acc[2] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[2],buffc[3], 8),buffw[1],acc[2]);
        acc[3] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[3],zerobuff, 8),buffw[1],acc[3]);
        acc[0] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[0],buffc[1],16),buffw[2],acc[0]);
        acc[1] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[1],buffc[2],16),buffw[2],acc[1]);
        acc[2] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[2],buffc[3],16),buffw[2],acc[2]);
        acc[3] = mac_4x8_8x4(aie::shuffle_down_fill(buffc[3],zerobuff,16),buffw[2],acc[3]);
      }

      // Apply RELU:
      *itw++ = aie::max(acc[0].to_vector<bfloat16>(),bfloat16(0));
      *itw++ = aie::max(acc[1].to_vector<bfloat16>(),bfloat16(0));
      *itw++ = aie::max(acc[2].to_vector<bfloat16>(),bfloat16(0));
      *itw++ = aie::max(acc[3].to_vector<bfloat16>(),bfloat16(0));
    } // oc
    idat += 16;                 // Advance by 1 input image row
  } // rr
}

// ------------------------------------------------------------
// Run
// ------------------------------------------------------------

void conv2d_w3::run( input_buffer      <bfloat16>& ifm_i,
                     input_async_buffer<bfloat16>& wts_i,
                     output_buffer     <bfloat16>& ofm_o )

{
  grab_wts( wts_i );
  filter_3x3( ifm_i, wts_i, ofm_o );
}


