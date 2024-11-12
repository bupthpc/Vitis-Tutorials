//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>

using namespace adf;

class dense_w7 {
private:
  const bfloat16 exp_S = 185.0f;
  const float    exp_B = 16256.0f;
public:
  unsigned fsm_state;
  alignas(32) bfloat16 data[1152];
  alignas(32) bfloat16 scratch[16];

  dense_w7( void );

  void grab_wts( input_async_buffer<bfloat16>& wts_i );

  void flatten( input_buffer<bfloat16>& ifm_i );

  void fully_connected( input_async_buffer<bfloat16>& wts_i );

  void softmax( output_buffer<bfloat16>& ofm_o );

  void run( input_buffer      <bfloat16>& ifm_i,
            input_async_buffer<bfloat16>& wts_i,
            output_buffer     <bfloat16>& ofm_o );

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( dense_w7::run );
  }
};

