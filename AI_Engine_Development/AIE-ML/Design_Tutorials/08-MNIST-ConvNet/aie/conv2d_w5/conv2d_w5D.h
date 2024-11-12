//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>

using namespace adf;

class conv2d_w5D {
public:
  unsigned state;
  alignas(32) bfloat16 data0[3*8*32];
  alignas(32) bfloat16 data1[3*8*32];
  alignas(32) bfloat16 data2[3*8*32];
  alignas(32) bfloat16 data3[3*8*32];

  conv2d_w5D( void );

  void grab_wts( input_async_buffer<bfloat16>& wts_i );

  void filter_3x3( input_buffer<bfloat16>& ifm_i, input_async_buffer<bfloat16>& wts_i );

  void pass_outputs( input_stream<bfloat16>* pass_i, output_buffer<bfloat16>& ofm_o );

  void run( input_buffer      <bfloat16>& ifm_i,
            input_async_buffer<bfloat16>& wts_i,
            input_stream<bfloat16>* pass_i,
            output_buffer<bfloat16>& ofm_o );

static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( conv2d_w5D::run );
  }
};

