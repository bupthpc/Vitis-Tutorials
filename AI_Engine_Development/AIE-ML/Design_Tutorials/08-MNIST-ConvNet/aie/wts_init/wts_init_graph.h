//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include "wts_init.h"

using namespace adf;

// ------------------------------------------------------------
// Top Level Graph
// ------------------------------------------------------------

template<class T, unsigned SIZE>
class wts_init_graph : public graph {
public:
  kernel      kk;
  input_port  wts_i;
  output_port wts_o;

  wts_init_graph( void )
  {
    kk = kernel::create_object<wts_init<T,SIZE> >();
    source(kk) = "wts_init.cpp";
    runtime<ratio>(kk) = 0.9;

    connect<>( wts_i,      kk.in[0] );
    connect<>( kk.out[0],  wts_o );
    dimensions(kk.out[0]) = {SIZE};
  }
};


