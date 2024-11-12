//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>

using namespace adf;

class conv2d_w1 {
public:
  unsigned fsm_state;

  conv2d_w1(void);

  void grab_wts( input_async_buffer<bfloat16>& wts_i );

  void passthru( input_buffer<bfloat16>& ifm_i, output_buffer<bfloat16>& ofm_o );

  void filter_3x3( input_buffer<bfloat16>&       ifm_i,
                   input_async_buffer<bfloat16>& wts_i,
                   output_buffer<bfloat16>&      ofm_o );

  void run( input_buffer      <bfloat16>& ifm_i,
            input_async_buffer<bfloat16>& wts_i,
            output_buffer     <bfloat16>& ofm_o );

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( conv2d_w1::run );
  }
};

