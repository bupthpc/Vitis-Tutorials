//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#include "tdm_mixer_graph.h"

// Instantiate AIE graph:
tdm_mixer_graph<64,32> aie_dut;

// Initialize and run the graph:
int main(void)
{
  aie_dut.init();
  aie_dut.run(4);
  aie_dut.end();

  return 0;
}
