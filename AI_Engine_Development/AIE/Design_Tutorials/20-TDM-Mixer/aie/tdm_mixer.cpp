//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#include <adf.h>
#include <aie_api/aie.hpp>

#include "tdm_mixer.h"


// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

template<unsigned NSAMP, unsigned CC>
tdm_mixer<NSAMP,CC>::tdm_mixer( int (&phase_inc_i)[CC] ) : phase_inc(phase_inc_i)
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
  // Initialize phase:
  auto it = aie::begin_vector<8>(phase);
  for (unsigned ii=0; ii < CC/8; ii++)
  {
    *it++ = aie::zeros<int,8>();
  } // ii
}

// ------------------------------------------------------------
// Run
// ------------------------------------------------------------

template<unsigned NSAMP, unsigned CC>
void tdm_mixer<NSAMP,CC>::run( input_buffer<cint16>& sig_i, output_buffer<cint16>& sig_o )
{
  auto itr = aie::begin_vector<8>(sig_i);
  auto itw = aie::begin_vector<8>(sig_o);
  aie::vector<cint16,8> ramp;

  // Loop over channels:
  for (unsigned cc=0; cc < CC; cc++)
    chess_prepare_for_pipelining
  {
    // Compute phase ramp for this channel:
    int step = 0;
    int incr = phase_inc[cc];
    for (unsigned ii=0; ii < 8; ii++) {
      ramp.set(aie::sincos_complex(step),ii);
      step += incr;
    }

    // Initialize this channel phase from last time:
    int curr = phase[cc];

    // Process this channel:
    for (unsigned ii=0; ii < NSAMP/8; ii++)
      chess_prepare_for_pipelining
    {
      auto data_i = *itr++;
      auto mixer  = (aie::mul(ramp,aie::sincos_complex(curr))).to_vector<cint16>(15);
      *itw++      = (aie::mul(data_i,mixer)).to_vector<cint16>(15);
      curr += step;
    } // ii

    // Save this channel phase for next time:
    phase[cc] = curr;
  } // cc
}


