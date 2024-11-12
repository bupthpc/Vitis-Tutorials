//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>

using namespace adf;

template<class T, unsigned SIZE>
class wts_init {
public:
  unsigned state;
  wts_init(void);

  // Run:
  void run( input_stream<T>* sig_i, output_async_buffer<T>& sig_o );

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( wts_init::run );
  }
};

