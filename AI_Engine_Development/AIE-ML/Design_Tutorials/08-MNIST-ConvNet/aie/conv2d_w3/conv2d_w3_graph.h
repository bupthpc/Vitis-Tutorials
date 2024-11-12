//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include "conv2d_w3.h"
#include "wts_init_graph.h"

using namespace adf;

// ------------------------------------------------------------
// Top Level Graph
// ------------------------------------------------------------

class conv2d_w3_graph : public graph {
public:
  input_port  ifm_i;
  input_port  wts_i;
  output_port ofm_o;

  kernel kk;
  wts_init_graph<bfloat16,9280> weights;
  shared_buffer<bfloat16>       MT0;
  shared_buffer<bfloat16>       MT1;

  conv2d_w3_graph( void )
  {
    kk = kernel::create_object<conv2d_w3>();
    source(kk) = "conv2d_w3.cpp";
    runtime<ratio>(kk) = 0.9;
    repetition_count(kk) = 4;

    tiling_parameters bdw0 = { .buffer_dimension    = {16*16*13}, // (layer,col,row)
                               .tiling_dimension    = {2},
                               .offset              = {0},
                               .tile_traversal      = {{.dimension=0,.stride=2,.wrap=8*16*13}} };

    tiling_parameters bdr0 = { .buffer_dimension    = {16,16,13}, // (layer,col,row)
                               .tiling_dimension    = { 8,16,1},
                               .offset              = {0,0,0},
                               .tile_traversal      = {{.dimension=0,.stride= 8,.wrap=2},
                                                       {.dimension=1,.stride=16,.wrap=1},
                                                       {.dimension=2,.stride=1,.wrap=13}} };

    MT0 = shared_buffer<bfloat16>::create({16*16*13},1,1);
    write_access(MT0.in[0]) = tiling(bdw0);
    read_access(MT0.out[0]) = tiling(bdr0);
    repetition_count(MT0) = 4;

    tiling_parameters bdw1 = { .buffer_dimension    = {64,16,11}, // (layer,col,row)
                               .tiling_dimension    = {4,4,1},
                               .offset              = {0,0,0},
                               .tile_traversal      = {{.dimension=1,.stride=4,.wrap=4},
                                                       {.dimension=0,.stride=4,.wrap=16},
                                                       {.dimension=2,.stride=1,.wrap=11}} };

    tiling_parameters bdr1 = { .buffer_dimension    = {64*16*11}, // (layer,col,row)
                               .tiling_dimension    = {2},
                               .offset              = {0},
                               .tile_traversal      = {{.dimension=0,.stride=2,.wrap=64*16*11/2}} };

    MT1 = shared_buffer<bfloat16>::create({64*16*11},1,1);
    write_access(MT1.in[0]) = tiling(bdw1);
    read_access(MT1.out[0]) = tiling(bdr1);
    repetition_count(MT1) = 4;

    connect<>( wts_i,            weights.wts_i );
    connect<>( weights.wts_o,    kk.in[1]  );    dimensions(kk.in[1])  = { 9280 };
    connect<>( ifm_i,            MT0.in[0] );
    connect<>( MT0.out[0],       kk.in[0]  );    dimensions(kk.in[0])  = { 13*16*16 };
    connect<>( kk.out[0],        MT1.in[0] );    dimensions(kk.out[0]) = { 11*16*64 };
    connect<>( MT1.out[0],       ofm_o     );
    single_buffer(kk.in[1]);
  }
};


