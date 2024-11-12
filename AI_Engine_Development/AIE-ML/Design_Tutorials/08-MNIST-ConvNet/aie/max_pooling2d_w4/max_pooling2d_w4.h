//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>

using namespace adf;

class max_pooling2d_w4 {
public:

  max_pooling2d_w4( void );

  void run( input_buffer<bfloat16>& ifm_i, output_buffer<bfloat16>& ofm_o );

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( max_pooling2d_w4::run );
  }
};

