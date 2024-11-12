//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include "conv2d_w1.h"
#include "wts_init_graph.h"

using namespace adf;

// ------------------------------------------------------------
// Top Level Graph
// ------------------------------------------------------------

class conv2d_w1_graph : public graph {
public:
  input_port  ifm_i;
  input_port  wts_i;
  output_port ofm_o;

  kernel kk;
  shared_buffer<bfloat16> MT;
  wts_init_graph<bfloat16,160> weights;

  conv2d_w1_graph( void )
  {
    kk = kernel::create_object<conv2d_w1>();
    source(kk) = "conv2d_w1.cpp";
    runtime<ratio>(kk) = 0.9;
    repetition_count(kk) = 4;

    // Note: <bfloat16> is 16-bit but memory tile requires 32-bit alignment:
    tiling_parameters bdw = { .buffer_dimension   = {28*28*4},
                              .tiling_dimension   = {2},
                              .offset             = {0},
                              .tile_traversal     = {{.dimension=0,.stride=2,.wrap=28*28*2}} };
    // Note: Zero-pad images by 4 pixels on the right side:
    tiling_parameters bdr = { .buffer_dimension   = {28,28,4},
                              .tiling_dimension   = {32,28,1},
                              .offset             = {0,0,0},
                              .tile_traversal     = {{.dimension=0,.stride=32,.wrap=1},
                                                     {.dimension=1,.stride=28,.wrap=1},
                                                     {.dimension=2,.stride= 1,.wrap=4}},
                              .boundary_dimension = {28,28,4} };

    MT = shared_buffer<bfloat16>::create({4*28*28},1,1);
    write_access(MT.in[0]) = tiling(bdw);
    read_access(MT.out[0]) = tiling(bdr);
    repetition_count(MT) = 1;

    connect<>( wts_i,          weights.wts_i );
    connect<>( weights.wts_o,  kk.in[1] );  dimensions(kk.in[1])  = {160};
    connect<>( ifm_i,          MT.in[0] );
    connect<>( MT.out[0],      kk.in[0] );  dimensions(kk.in[0])  = {32*28};
    connect<>( kk.out[0],      ofm_o );     dimensions(kk.out[0]) = {32*26*16};
    single_buffer(kk.in[1]);
  }
};


