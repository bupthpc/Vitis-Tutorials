//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#include <adf.h>
#include <aie_api/aie.hpp>

#include "wts_init.h"
#include <adf/x86sim/x86simDebug.h>

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

template<class T, unsigned SIZE>
wts_init<T,SIZE>::wts_init(void)
{
  aie::set_rounding(aie::rounding_mode::symmetric_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
  state = 0;
}

// ------------------------------------------------------------
// Run
// ------------------------------------------------------------

template<class T, unsigned SIZE>
void wts_init<T,SIZE>::run( input_stream<T>* sig_i, output_async_buffer<T>& sig_o )
{
  aie::vector<T,16> vec;
  // This will only happen once:
  if (state == 0) {
    sig_o.acquire();
    auto itw = aie::begin_vector<16>(sig_o);
    for (unsigned ii=0; ii < SIZE/16; ii++)
      chess_prepare_for_pipelining
    {
      vec.insert(0,readincr_v<8>(sig_i));
      vec.insert(1,readincr_v<8>(sig_i));
      *itw++ = vec;
    }
    sig_o.release();
    state = 1;
  }
  X86SIM_TERMINATE_CURRENT_THREAD;
}


