//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include "conv2d_w1_graph.h"
#include "max_pooling2d_w2_graph.h"
#include "conv2d_w3_graph.h"
#include "max_pooling2d_w4_graph.h"
#include "conv2d_w5_graph.h"
#include "dense_w7_graph.h"

using namespace adf;

// ------------------------------------------------------------
// Top Level Graph
// ------------------------------------------------------------

class mnist_graph : public graph {
public:
  input_port  ifm_i;
  output_port ofm_o;

  input_port                wts1_i;
  input_port                wts3_i;
  std::array<input_port,4>  wts5_i;
  input_port                wts7_i;

  // Graphs:
  conv2d_w1_graph        layer_w1;
  max_pooling2d_w2_graph layer_w2;
  conv2d_w3_graph        layer_w3;
  max_pooling2d_w4_graph layer_w4;
  conv2d_w5_graph        layer_w5;
  dense_w7_graph         layer_w7;

  // Constructor:
  mnist_graph( void )
  {
    connect<>( ifm_i,             layer_w1.ifm_i );
    connect<>( layer_w1.ofm_o,    layer_w2.ifm_i );
    connect<>( layer_w2.ofm_o,    layer_w3.ifm_i );
    connect<>( layer_w3.ofm_o,    layer_w4.ifm_i );
    connect<>( layer_w4.ofm_o,    layer_w5.ifm_i );
    connect<>( layer_w5.ofm_o,    layer_w7.ifm_i );
    connect<>( layer_w7.ofm_o,    ofm_o );

    connect<>( wts1_i,    layer_w1.wts_i );
    connect<>( wts3_i,    layer_w3.wts_i );
    connect<>( wts5_i[0], layer_w5.wts_i[0] );
    connect<>( wts5_i[1], layer_w5.wts_i[1] );
    connect<>( wts5_i[2], layer_w5.wts_i[2] );
    connect<>( wts5_i[3], layer_w5.wts_i[3] );
    connect<>( wts7_i,    layer_w7.wts_i );
  }
};


