//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#include "conv2d_w5_graph.h"

class dut_graph : public graph {
public:
  conv2d_w5_graph                                    dut;
  input_plio                                         ifm_i;
  std::array<input_plio,4>                           wts_i;
  output_plio                                        ofm_o;

  dut_graph( void )
  {
    ifm_i    =  input_plio::create("PLIO_i", plio_64_bits,"data/ifm_i.txt");
    wts_i[0] =  input_plio::create("PLIO_w0",plio_64_bits,"data/wts_0_i.txt");
    wts_i[1] =  input_plio::create("PLIO_w1",plio_64_bits,"data/wts_1_i.txt");
    wts_i[2] =  input_plio::create("PLIO_w2",plio_64_bits,"data/wts_2_i.txt");
    wts_i[3] =  input_plio::create("PLIO_w3",plio_64_bits,"data/wts_3_i.txt");
    ofm_o    = output_plio::create("PLIO_o", plio_64_bits,"data/ofm_o.txt");


    connect<>(wts_i[0].out[0],   dut.wts_i[0] );
    connect<>(wts_i[1].out[0],   dut.wts_i[1] );
    connect<>(wts_i[2].out[0],   dut.wts_i[2] );
    connect<>(wts_i[3].out[0],   dut.wts_i[3] );

    connect<>(ifm_i.out[0],      dut.ifm_i );
    connect<>(dut.ofm_o,         ofm_o.in[0]);

    location<kernel>(dut.kkA) = tile(18,1);
    location<kernel>(dut.kkB) = tile(19,1);
    location<kernel>(dut.kkC) = tile(20,1);
    location<kernel>(dut.kkD) = tile(21,1);

    location<kernel>(dut.weights[0].kk) = tile(18,0);
    location<kernel>(dut.weights[1].kk) = tile(19,0);
    location<kernel>(dut.weights[2].kk) = tile(20,0);
    location<kernel>(dut.weights[3].kk) = tile(21,0);
  }
};

// Instantiate AIE graph:
dut_graph aie_dut;

// Initialize and run the graph:
int main(void)
{
  aie_dut.init();
  aie_dut.run(1);               // 1 iteration = 4 images
  aie_dut.end();

  return 0;
}
