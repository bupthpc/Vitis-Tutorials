//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Author: Mark Rollins

#pragma once

#include <adf.h>
#include "tdm_mixer.h"

using namespace adf;

// ------------------------------------------------------------
// Top Level Graph
// ------------------------------------------------------------

template<unsigned NSAMP, unsigned CC>
class tdm_mixer_graph : public graph {
private:
  kernel kk;

public:
  input_plio  sig_i;
  output_plio sig_o;

#include "tdm_mixer_phase_inc.h"

  tdm_mixer_graph( void )
  {
    sig_i =  input_plio::create("PLIO_i", plio_64_bits, "data/sig_i.txt" );
    sig_o = output_plio::create("PLIO_o", plio_64_bits, "data/sig_o.txt" );

    // We have two versions to consider:
    //    tdm_mixer.cpp      --> baseline initial version (not optimized)
    //    tdm_mixer_fast.cpp --> optimized version
    kk = kernel::create_object<tdm_mixer<NSAMP,CC> >(phase_inc);
    source(kk) = "tdm_mixer_fast.cpp";
    runtime<ratio>(kk) = 0.9;

    // Input stream is TDM channels (one sample from each channel in turn):
    tiling_parameters bdw = { .buffer_dimension       = {NSAMP,CC},
                              .tiling_dimension       = {1,1},
                              .offset                 = {0,0},
                              .tile_traversal         = {{.dimension=1,.stride=1,.wrap=CC},
                                                         {.dimension=0,.stride=1,.wrap=NSAMP}} };

    // Output stream restores TDM order (written in channel order from core):
    tiling_parameters bdr = { .buffer_dimension       = {NSAMP,CC},
                              .tiling_dimension       = {1,1},
                              .offset                 = {0,0},
                              .tile_traversal         = {{.dimension=1,.stride=1,.wrap=CC},
                                                         {.dimension=0,.stride=1,.wrap=NSAMP}} };
    // Assign tiling parameters:
    write_access(kk.in[0]) = tiling(bdw);
    read_access(kk.out[0]) = tiling(bdr);

    // Connections:
    connect<>( sig_i.out[0],      kk.in[0] );   dimensions(kk.in[0] ) = {NSAMP * CC};
    connect<>( kk.out[0],      sig_o.in[0] );   dimensions(kk.out[0]) = {NSAMP * CC};
  }
};


