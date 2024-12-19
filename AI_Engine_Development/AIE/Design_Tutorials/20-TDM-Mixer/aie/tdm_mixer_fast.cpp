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
  std::array<aie::restrict_vector_iterator<cint16,8,aie_dm_resource::none>,4>
    itr = { aie::begin_restrict_vector<8,aie_dm_resource::none>(sig_i),
            aie::begin_restrict_vector<8,aie_dm_resource::none>(sig_i),
            aie::begin_restrict_vector<8,aie_dm_resource::none>(sig_i),
            aie::begin_restrict_vector<8,aie_dm_resource::none>(sig_i) };

  std::array<aie::restrict_vector_iterator<cint16,8,aie_dm_resource::none>,4>
    itw = { aie::begin_restrict_vector<8,aie_dm_resource::none>(sig_o),
            aie::begin_restrict_vector<8,aie_dm_resource::none>(sig_o),
            aie::begin_restrict_vector<8,aie_dm_resource::none>(sig_o),
            aie::begin_restrict_vector<8,aie_dm_resource::none>(sig_o) };

  std::array<aie::vector<cint16,8>,4> ramp;
  std::array<aie::vector<cint16,8>,4> data;
  std::array<aie::vector<cint16,8>,4> mixer;

  // Advance to start of each channel:
  itr[1] +=   NSAMP/8;   itw[1] +=   NSAMP/8;
  itr[2] += 2*NSAMP/8;   itw[2] += 2*NSAMP/8;
  itr[3] += 3*NSAMP/8;   itw[3] += 3*NSAMP/8;

  // Loop over channels:
  for (unsigned cc=0; cc < CC; cc+=4)
    chess_prepare_for_pipelining
  {
    // Compute phase ramp for this channel:
    std::array<int,4> step = {0,0,0,0};
    std::array<int,4> incr = { phase_inc[cc], phase_inc[cc+1], phase_inc[cc+2], phase_inc[cc+3] };
    for (unsigned ii=0; ii < 8; ii++) {
      ramp[0].set(aie::sincos_complex(step[0]),ii);
      ramp[1].set(aie::sincos_complex(step[1]),ii);
      ramp[2].set(aie::sincos_complex(step[2]),ii);
      ramp[3].set(aie::sincos_complex(step[3]),ii);
      step[0] += incr[0];
      step[1] += incr[1];
      step[2] += incr[2];
      step[3] += incr[3];
    }

    // Initialize this channel phase from last time:
    std::array<int,4> curr = { phase[cc], phase[cc+1], phase[cc+2], phase[cc+3] };

    // Process this channel:
    for (unsigned ii=0; ii < NSAMP/8; ii++)
      chess_prepare_for_pipelining
    {
      data[0] = *itr[0]++;
      data[1] = *itr[1]++;
      data[2] = *itr[2]++;
      data[3] = *itr[3]++;
      mixer[0] = (aie::mul(ramp[0],aie::sincos_complex(curr[0]))).to_vector<cint16>(15);
      mixer[1] = (aie::mul(ramp[1],aie::sincos_complex(curr[1]))).to_vector<cint16>(15);
      mixer[2] = (aie::mul(ramp[2],aie::sincos_complex(curr[2]))).to_vector<cint16>(15);
      mixer[3] = (aie::mul(ramp[3],aie::sincos_complex(curr[3]))).to_vector<cint16>(15);
      *itw[0]++ = (aie::mul(data[0],mixer[0])).to_vector<cint16>(15);
      *itw[1]++ = (aie::mul(data[1],mixer[1])).to_vector<cint16>(15);
      *itw[2]++ = (aie::mul(data[2],mixer[2])).to_vector<cint16>(15);
      *itw[3]++ = (aie::mul(data[3],mixer[3])).to_vector<cint16>(15);
      curr[0] += step[0];
      curr[1] += step[1];
      curr[2] += step[2];
      curr[3] += step[3];
    } // ii

    // Update pointers for next 4 channels:
    itr[0] += 3*NSAMP/8;   itw[0] += 3*NSAMP/8;
    itr[1] += 3*NSAMP/8;   itw[1] += 3*NSAMP/8;
    itr[2] += 3*NSAMP/8;   itw[2] += 3*NSAMP/8;
    itr[3] += 3*NSAMP/8;   itw[3] += 3*NSAMP/8;

    // Save this channel phase for next time:
    phase[cc]   = curr[0];
    phase[cc+1] = curr[1];
    phase[cc+2] = curr[2];
    phase[cc+3] = curr[3];
  } // cc
}


