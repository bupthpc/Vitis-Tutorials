//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include "conv2d_w5A.h"
#include "conv2d_w5B.h"
#include "conv2d_w5C.h"
#include "conv2d_w5D.h"
#include "wts_init_graph.h"

using namespace adf;

// ------------------------------------------------------------
// Top Level Graph
// ------------------------------------------------------------

class conv2d_w5_graph : public graph {
public:
  input_port                ifm_i;
  std::array<input_port,4>  wts_i;
  output_port               ofm_o;

  kernel kkA;
  kernel kkB;
  kernel kkC;
  kernel kkD;
  std::array<wts_init_graph<bfloat16,18464>,4> weights;
  shared_buffer<bfloat16>                      MT0;
  shared_buffer<bfloat16>                      MT1;

  conv2d_w5_graph( void )
  {
    kkA = kernel::create_object<conv2d_w5A>();
    kkB = kernel::create_object<conv2d_w5B>();
    kkC = kernel::create_object<conv2d_w5C>();
    kkD = kernel::create_object<conv2d_w5D>();
    source(kkA) = "conv2d_w5A.cpp";
    source(kkB) = "conv2d_w5B.cpp";
    source(kkC) = "conv2d_w5C.cpp";
    source(kkD) = "conv2d_w5D.cpp";
    runtime<ratio>(kkA) = 0.9;
    runtime<ratio>(kkB) = 0.9;
    runtime<ratio>(kkC) = 0.9;
    runtime<ratio>(kkD) = 0.9;
    repetition_count(kkA) = 4;
    repetition_count(kkB) = 4;
    repetition_count(kkC) = 4;
    repetition_count(kkD) = 4;

    tiling_parameters bdw0 = { .buffer_dimension       = {64*8*5}, // (layer,col,row)
                               .tiling_dimension       = {2},
                               .offset                 = {0},
                               .tile_traversal         = {{.dimension=0,.stride=2,.wrap=32*8*5}} };

    tiling_parameters bdr0 = { .buffer_dimension       = {64,8,5}, // (layer,col,row)
                               .tiling_dimension       = {8,8,1},
                               .offset                 = {0,0,0},
                               .tile_traversal         = {{.dimension=0,.stride= 8,.wrap=8},
                                                          {.dimension=1,.stride= 8,.wrap=1},
                                                          {.dimension=2,.stride= 1,.wrap=5}} };

    MT0 = shared_buffer<bfloat16>::create({64*8*5},1,1);
    write_access(MT0.in[0]) = tiling(bdw0);
    read_access(MT0.out[0]) = tiling(bdr0);
    repetition_count(MT0) = 4;

    tiling_parameters bdw1 = { .buffer_dimension       = {128,8,3}, // (layer,col,row)
                               .tiling_dimension       = {4,8,1},
                               .offset                 = {0,0,0},
                               .tile_traversal         = {{.dimension=0,.stride=4,.wrap=32},
                                                          {.dimension=1,.stride=8,.wrap=1},
                                                          {.dimension=2,.stride=1,.wrap=3}} };

    tiling_parameters bdr1 = { .buffer_dimension       = {128*8*3}, // (layer,col,row)
                               .tiling_dimension       = {2},
                               .offset                 = {0},
                               .tile_traversal         = {{.dimension=0,.stride=2,.wrap=64*8*3}} };

    MT1 = shared_buffer<bfloat16>::create({128*8*3},1,1);
    write_access(MT1.in[0]) = tiling(bdw1);
    read_access(MT1.out[0]) = tiling(bdr1);
    repetition_count(MT1) = 4;

    connect<>(wts_i[0],   weights[0].wts_i );
    connect<>(wts_i[1],   weights[1].wts_i );
    connect<>(wts_i[2],   weights[2].wts_i );
    connect<>(wts_i[3],   weights[3].wts_i );

    connect<>( weights[0].wts_o,       kkA.in[1] );       dimensions(kkA.in[1]) = { 18464 };
    connect<>( weights[1].wts_o,       kkB.in[1] );       dimensions(kkB.in[1]) = { 18464 };
    connect<>( weights[2].wts_o,       kkC.in[1] );       dimensions(kkC.in[1]) = { 18464 };
    connect<>( weights[3].wts_o,       kkD.in[1] );       dimensions(kkD.in[1]) = { 18464 };

    connect<>( ifm_i,          MT0.in[0] );
    connect<>( MT0.out[0],     kkA.in[0] );       dimensions(kkA.in[0])  = { 5*8*64 };
    connect<>( MT0.out[0],     kkB.in[0] );       dimensions(kkB.in[0])  = { 5*8*64 };
    connect<>( MT0.out[0],     kkC.in[0] );       dimensions(kkC.in[0])  = { 5*8*64 };
    connect<>( MT0.out[0],     kkD.in[0] );       dimensions(kkD.in[0])  = { 5*8*64 };

    connect<>( kkA.out[0],     kkB.in[2] );
    connect<>( kkB.out[0],     kkC.in[2] );
    connect<>( kkC.out[0],     kkD.in[2] );
    connect<>( kkD.out[0],     MT1.in[0] );       dimensions(kkD.out[0]) = { 3*8*128 };
    connect<>( MT1.out[0],     ofm_o     );

    single_buffer(kkA.in[1]);
    single_buffer(kkB.in[1]);
    single_buffer(kkC.in[1]);
    single_buffer(kkD.in[1]);
  }
};


