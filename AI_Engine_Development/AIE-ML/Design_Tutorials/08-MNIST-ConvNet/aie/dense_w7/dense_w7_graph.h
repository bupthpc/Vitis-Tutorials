//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include "dense_w7.h"
#include "wts_init_graph.h"

using namespace adf;

// ------------------------------------------------------------
// Top Level Graph
// ------------------------------------------------------------

class dense_w7_graph : public graph {
public:
  input_port  ifm_i;
  input_port  wts_i;
  output_port ofm_o;

  kernel kk;
  wts_init_graph<bfloat16,11536> weights;

  dense_w7_graph( void )
  {
    kk = kernel::create_object<dense_w7>();
    source(kk) = "dense_w7.cpp";
    runtime<ratio>(kk) = 0.9;
    repetition_count(kk) = 4;

    connect<>( wts_i,               weights.wts_i );
    connect<>( weights.wts_o,       kk.in[1] );        dimensions(kk.in[1]) = {11536}; // 1152*10+10 = 11530, not a power of 16
    connect<>( ifm_i,               kk.in[0] );        dimensions(kk.in[0]) = {3*8*128};
    connect<>( kk.out[0],           ofm_o );           dimensions(kk.out[0]) = {16};
    single_buffer(kk.in[1]);
  }
};


