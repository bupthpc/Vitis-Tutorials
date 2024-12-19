//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>

using namespace adf;

template<unsigned NSAMP,unsigned CC>
class tdm_mixer {
public:
  tdm_mixer( int (&phase_inc_i)[CC] );

  alignas(16) int phase[CC];
  alignas(16) int (&phase_inc)[CC];

  // Run:
  void run( input_buffer<cint16>& sig_i, output_buffer<cint16>& sig_o );

  static void registerKernelClass( void )
  {
    REGISTER_FUNCTION( tdm_mixer::run );
    REGISTER_PARAMETER( phase_inc );
  }
};

