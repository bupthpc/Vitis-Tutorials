//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#include "max_pooling2d_w2_graph.h"

class dut_graph : public graph {
public:
  max_pooling2d_w2_graph dut;
  input_plio   ifm_i;
  output_plio  ofm_o;

  dut_graph( void )
  {
    ifm_i =  input_plio::create("PLIO_i",plio_64_bits,"data/ifm_i.txt");
    ofm_o = output_plio::create("PLIO_o",plio_64_bits,"data/ofm_o.txt");

    connect<>(ifm_i.out[0], dut.ifm_i);
    connect<>(dut.ofm_o,    ofm_o.in[0]);
  }
};

// Instantiate AIE graph:
dut_graph aie_dut;

// Initialize and run the graph:
int main(void)
{
  aie_dut.init();
  aie_dut.run(1); // 1 iteration = 4 images
  aie_dut.end();

  return 0;
}
