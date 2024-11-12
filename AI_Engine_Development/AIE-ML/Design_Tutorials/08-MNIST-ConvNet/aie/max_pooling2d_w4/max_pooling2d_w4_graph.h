//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include "max_pooling2d_w4.h"

using namespace adf;

// ------------------------------------------------------------
// Top Level Graph
// ------------------------------------------------------------

class max_pooling2d_w4_graph : public graph {
public:
  input_port  ifm_i;
  output_port ofm_o;

  kernel kk;

  // Constructor:
  max_pooling2d_w4_graph( void )
  {
    // Create kernel:
    kk = kernel::create_object<max_pooling2d_w4>();
    source(kk) = "max_pooling2d_w4.cpp";
    runtime<ratio>(kk) = 0.9;
    repetition_count(kk) = 4;

    connect<>( ifm_i,       kk.in[0] );  dimensions(kk.in[0])  = {64*11*16};
    connect<>( kk.out[0],   ofm_o );     dimensions(kk.out[0]) = {64* 5*8};
  }
};


