//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#include "mnist_graph.h"
#include "num_iter.h"

class dut_graph : public graph {
public:
  input_plio               ifm_i;
  input_plio               wts1_i;
  input_plio               wts3_i;
  input_plio               wts7_i;
  output_plio              ofm_o;
  std::array<input_plio,4> wts5_i;
  mnist_graph              dut;

  dut_graph( void )
  {
    ifm_i     =  input_plio::create("PLIO_i",     plio_64_bits, "data/ifm_i.txt");
    wts1_i    =  input_plio::create("PLIO_wts1",  plio_64_bits, "data/wts1_i.txt");
    wts3_i    =  input_plio::create("PLIO_wts3",  plio_64_bits, "data/wts3_i.txt");
    wts5_i[0] =  input_plio::create("PLIO_wts5_0",plio_64_bits, "data/wts5_0_i.txt");
    wts5_i[1] =  input_plio::create("PLIO_wts5_1",plio_64_bits, "data/wts5_1_i.txt");
    wts5_i[2] =  input_plio::create("PLIO_wts5_2",plio_64_bits, "data/wts5_2_i.txt");
    wts5_i[3] =  input_plio::create("PLIO_wts5_3",plio_64_bits, "data/wts5_3_i.txt");
    wts7_i    =  input_plio::create("PLIO_wts7",  plio_64_bits, "data/wts7_i.txt");
    ofm_o     = output_plio::create("PLIO_o",     plio_64_bits, "data/ofm_o.txt");

    connect<>( ifm_i.out[0],     dut.ifm_i    );
    connect<>( dut.ofm_o,        ofm_o.in[0]  );
    connect<>( wts1_i.out[0],    dut.wts1_i   );
    connect<>( wts3_i.out[0],    dut.wts3_i   );
    connect<>( wts7_i.out[0],    dut.wts7_i   );
    connect<>( wts5_i[0].out[0], dut.wts5_i[0] );
    connect<>( wts5_i[1].out[0], dut.wts5_i[1] );
    connect<>( wts5_i[2].out[0], dut.wts5_i[2] );
    connect<>( wts5_i[3].out[0], dut.wts5_i[3] );

    location<kernel>(dut.layer_w1.kk)                   = tile(18,1);

    location<kernel>(dut.layer_w1.weights.kk)           = tile(18,0);
    location<stack >(dut.layer_w1.weights.kk)           = bank(18,0,0);
    location<buffer>(dut.layer_w1.kk.in[1])             = bank(18,0,0);
    location<stack >(dut.layer_w1.kk)                   = bank(18,0,1);
    location<buffer>(dut.layer_w1.kk.in[0])             = { bank(18,0,2), bank(18,0,3) };

    location<stack >(dut.layer_w2.kk)                   = bank(19,0,0);
    location<stack >(dut.layer_w3.weights.kk)           = bank(19,0,1);

    location<kernel>(dut.layer_w2.kk)                   = tile(19,1);
    location<buffer>(dut.layer_w2.kk.out[0])            = { bank(19,1,0), bank(19,1,1) };
    location<stack >(dut.layer_w3.kk)                   = bank(19,1,2);

    location<kernel>(dut.layer_w3.kk)                   = tile(20,1);
    location<buffer>(dut.layer_w3.kk.out[0])            = { bank(20,1,0), bank(20,1,1) };

    location<kernel>(dut.layer_w3.weights.kk)           = tile(20,0);
    location<buffer>(dut.layer_w3.kk.in[0])             = { bank(20,0,0), bank(20,0,1) };
    location<buffer>(dut.layer_w3.kk.in[1])             = bank(20,0,2);

    location<kernel>(dut.layer_w4.kk)                   = tile(21,1);
    location<buffer>(dut.layer_w4.kk.out[0])            = { bank(21,1,0), bank(21,1,1) };
    location<stack >(dut.layer_w4.kk)                   = bank(21,1,2);

    location<buffer>(dut.layer_w4.kk.in[0])             = { bank(21,0,0), bank(21,0,2) };

    location<kernel>(dut.layer_w5.kkA)                  = tile(22,1);
    location<buffer>(dut.layer_w5.kkA.in[0])            = { bank(22,1,0), bank(22,1,1) };
    location<stack >(dut.layer_w5.kkA)                  = bank(22,1,2);

    location<kernel>(dut.layer_w5.weights[0].kk)        = tile(22,0);
    location<buffer>(dut.layer_w5.kkA.in[1])            = bank(22,0,0);
    location<stack >(dut.layer_w5.weights[0].kk)        = bank(22,0,3);

    location<kernel>(dut.layer_w5.kkB)                  = tile(23,1);
    location<buffer>(dut.layer_w5.kkB.in[0])            = { bank(23,1,0), bank(23,1,1) };
    location<stack >(dut.layer_w5.kkB)                  = bank(23,1,2);

    location<kernel>(dut.layer_w5.weights[1].kk)        = tile(23,0);
    location<buffer>(dut.layer_w5.kkB.in[1])            = bank(23,0,0);
    location<stack >(dut.layer_w5.weights[1].kk)        = bank(23,0,3);

    location<kernel>(dut.layer_w5.kkC)                  = tile(24,1);
    location<buffer>(dut.layer_w5.kkC.in[0])            = { bank(24,1,0), bank(24,1,1) };
    location<stack >(dut.layer_w5.kkC)                  = bank(24,1,2);

    location<kernel>(dut.layer_w5.weights[2].kk)        = tile(24,0);
    location<buffer>(dut.layer_w5.kkC.in[1])            = bank(24,0,0);
    location<stack >(dut.layer_w5.weights[2].kk)        = bank(24,0,3);

    location<kernel>(dut.layer_w5.kkD)                  = tile(25,1);
    location<buffer>(dut.layer_w5.kkD.in[0])            = { bank(25,1,0), bank(25,1,1) };
    location<buffer>(dut.layer_w5.kkD.out[0])           = { bank(25,1,2), bank(25,1,3) };

    location<kernel>(dut.layer_w5.weights[3].kk)        = tile(25,0);
    location<buffer>(dut.layer_w5.kkD.in[1])            = bank(25,0,0);
    location<stack >(dut.layer_w5.weights[3].kk)        = bank(25,0,3);
    location<stack >(dut.layer_w5.kkD)                  = bank(25,0,3);

    location<kernel>(dut.layer_w7.kk)                   = tile(26,1);
    location<buffer>(dut.layer_w7.kk.in[0])             = { bank(26,1,0), bank(26,1,1) };
    location<buffer>(dut.layer_w7.kk.out[0])            = { bank(26,1,2), bank(26,1,3) };

    location<kernel>(dut.layer_w7.weights.kk)           = tile(26,0);
    location<buffer>(dut.layer_w7.kk.in[1])             = bank(26,0,0);
    location<stack >(dut.layer_w7.weights.kk)           = bank(26,0,2);
    location<stack >(dut.layer_w7.kk)                   = bank(26,0,3);
  }
};

dut_graph aie_dut;

// Initialize and run the graph:
int main(void)
{
  aie_dut.init();
  aie_dut.run(NUM_ITER);               // 1 iteration = 4 images
  aie_dut.end();

  return 0;
}
