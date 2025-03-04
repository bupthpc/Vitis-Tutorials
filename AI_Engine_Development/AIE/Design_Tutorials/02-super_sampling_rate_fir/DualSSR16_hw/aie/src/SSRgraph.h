/*
Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
*/

#pragma once

#include <adf.h>
#include "system_settings.h"
#include "aie_kernels.h"


std::vector<cint16> taps = std::vector<cint16>({
    {   -82,  -253},{     0,  -204},{    11,   -35},{  -198,   273},
    {  -642,   467},{ -1026,   333},{  -927,     0},{  -226,   -73},
    {   643,   467},{   984,  1355},{   550,  1691},{     0,   647},
    {   538, -1656},{  2860, -3936},{  6313, -4587},{  9113, -2961},
    {  9582,     0},{  7421,  2411},{  3936,  2860},{  1023,  1409},
    {  -200,  -615},{     0, -1778},{   517, -1592},{   467,  -643},
    {  -192,   140},{  -882,   287},{ -1079,     0},{  -755,  -245},
    {  -273,  -198},{    22,    30},{    63,   194},{     0,   266}
});

std::vector<cint16> taps_aie(taps.rbegin(),taps.rend());

#define PHASE(N,NP) {taps_aie[N],taps_aie[N+NP],taps_aie[N+2*NP],taps_aie[N+3*NP]}

std::vector<cint16> taps8p[8] = { PHASE(0,8), PHASE(1,8), PHASE(2,8), PHASE(3,8),
    PHASE(4,8), PHASE(5,8), PHASE(6,8), PHASE(7,8) };

    const int SHIFT = 0; // to analyze the output generated by impulses at the input
    //const int SHIFT = 15; // for realistic input samples


    using namespace adf;

    // #define FirstCol 23
    // #define LastCol (FirstCol+7)

template<int COL>
    class FIRGraph_SSR8: public adf::graph
    {
    private:
        kernel k[8][8];

    public:
        input_port in[16]; // 8 columns, 2 streams per kernel
        output_port out[16]; // 8 columns, 2 streams per kernel

        FIRGraph_SSR8()
        {
            // k[N][0] is always the first in the cascade stream
            // Topology of the TopGraph
            //
            //      7,7   ...   ...   7,0 <--
            //  --> 6,0   ...   ...   6,7
            //       .     .     .     .
            //       .     .     .     .
            //       .     .     .     .
            //      1,7   ...   ...   1,0 <--
            //  --> 0,0   ...   ...   0,7

            const int NPhases = 8;
            const int FirstCol = COL;
            const int LastCol = COL + 7;

            for(int row = 0;row<NPhases; row++)
            for(int col=0;col<NPhases; col++)
            {
                int col1 = (row%2?NPhases-col-1:col); // Revert col order on odd rows

                // Which Phase in (row,col) ?
                int PhaseSelect = col-row;
                if(PhaseSelect<0) PhaseSelect += NPhases;
                int DiscardSample = (row>col?1:0); // Must discard 1 sample on some aie_kernels

                if(DiscardSample == 1) // Don't know how to make DiscardSample a constant suitable for a template parameter
                {
                    // kernel instanciation
                    if(col1==0)
                    {
                        k[row][col1] = kernel::create_object<DoubleStream::FIR_MultiKernel_cout<NUM_SAMPLES,SHIFT,true,false>>(taps8p[PhaseSelect]);
                    }
                    else
                    {
                        if(col1==NPhases-1)
                        {
                            k[row][col1] = kernel::create_object<DoubleStream::FIR_MultiKernel_cin<NUM_SAMPLES,SHIFT,true,false>>(taps8p[PhaseSelect]);
                        }
                        else
                        {
                            k[row][col1] = kernel::create_object<DoubleStream::FIR_MultiKernel_cincout<NUM_SAMPLES,SHIFT,true,false>>(taps8p[PhaseSelect]);
                        }
                    }
                }
                else
                {
                    // kernel instanciation
                    if(col1==0)
                    {
                        k[row][col1] = kernel::create_object<DoubleStream::FIR_MultiKernel_cout<NUM_SAMPLES,SHIFT,false,false>>(taps8p[PhaseSelect]);
                    }
                    else
                    {
                        if(col1==NPhases-1)
                        {
                            k[row][col1] = kernel::create_object<DoubleStream::FIR_MultiKernel_cin<NUM_SAMPLES,SHIFT,false,false>>(taps8p[PhaseSelect]);
                        }
                        else
                        {
                            k[row][col1] = kernel::create_object<DoubleStream::FIR_MultiKernel_cincout<NUM_SAMPLES,SHIFT,false,false>>(taps8p[PhaseSelect]);
                        }
                    }
                }
            }

            for(int i=0;i<NPhases;i++)
            for(int j=0;j<NPhases;j++)
            {
                runtime<ratio>(k[i][j]) = 0.9;
                source(k[i][j]) = "aie_kernels/FirDoubleStream.cpp";
                headers(k[i][j]) = {"aie_kernels/FirDoubleStream.h"};
            }

            // Constraints: location of the first kernel in the cascade
            for(int i=0;i<NPhases;i++)
            {
                int j = (i%2?LastCol:FirstCol); // 23 on even rows and 30on odd rows
                location<kernel>(k[i][0]) = tile(j,i);
            }


            // Cascade Connections
            for(int row=0;row<NPhases;row++)
            {
                for(int i=0;i<NPhases-1;i++) connect<cascade> (k[row][i].out[0],k[row][i+1].in[2]);
                connect<stream> (k[row][NPhases-1].out[0],out[2*row]);
                connect<stream> (k[row][NPhases-1].out[1],out[2*row+1]);
            }

            // Input Streams connections and FIFO constraint
            // std::array<connect<stream> *,16> net;

            for(int row = 0;row<NPhases;row++)
            for(int col=0;col<NPhases;col++)
            {
                int col1 = (row%2?NPhases-col-1:col); // kernel col is inverted on odd rows
                int fiforow = row%2;

                connect<stream> n0(in[2*col],k[row][col1].in[0]);
                connect<stream> n1(in[2*col+1],k[row][col1].in[1]);

#define FIFO_SETTINGS 2

#if FIFO_SETTINGS == 1
                //FIFO Length Estimation Proposal
                fifo_depth(n0) = 256;
                fifo_depth(n1) = 256;

                location<fifo>(n0) = dma_fifo(aie_tile, FirstCol+col, fiforow, 0x0000, 256);
                location<fifo>(n1) = dma_fifo(aie_tile, FirstCol+col, fiforow, 0x2000, 256);
#elif FIFO_SETTINGS == 2
                // Optimized FIFO Length
                #define LAT_INC 16
                int depth;

                if(!fiforow)    depth = (col)*LAT_INC;  // even rows
                else            depth = (col1)*LAT_INC; // odd rows

                if(depth>0)
                {
                    fifo_depth(n0) = depth;
                    fifo_depth(n1) = depth;

                    if(depth>32) // DMA FIFO sharing
                    {
                        location<fifo>(n0) = dma_fifo(aie_tile, FirstCol+col, fiforow, 0x0000, depth);
                        location<fifo>(n1) = dma_fifo(aie_tile, FirstCol+col, fiforow, 0x2000, depth);
                    }
                }
#endif

            }
        };
    };


template<int COL>
    class TopGraph: public adf::graph
    {
    public:
        FIRGraph_SSR8<COL> G;

        input_plio plin[16]; // 8 columns, 2 streams per kernel
        output_plio plout[16]; // 8 columns, 2 streams per kernel

        TopGraph()
        {
            for(int i=0;i<8;i++)
            {
                plin[2*i]   = input_plio::create("PhaseIn_"+std::to_string(i)+"_0_"+std::to_string(COL),plio_128_bits,"data/PhaseIn_"+std::to_string(i)+"_0_"+std::to_string(COL)+".txt",250);
                plin[2*i+1] = input_plio::create("PhaseIn_"+std::to_string(i)+"_1_"+std::to_string(COL),plio_128_bits,"data/PhaseIn_"+std::to_string(i)+"_1_"+std::to_string(COL)+".txt",250);

                plout[2*i]   = output_plio::create("PhaseOut_"+std::to_string(i)+"_0_"+std::to_string(COL),plio_128_bits,"data/PhaseOut_"+std::to_string(i)+"_0_"+std::to_string(COL)+".txt",250);
                plout[2*i+1] = output_plio::create("PhaseOut_"+std::to_string(i)+"_1_"+std::to_string(COL),plio_128_bits,"data/PhaseOut_"+std::to_string(i)+"_1_"+std::to_string(COL)+".txt",250);

                connect<> (plin[2*i].out[0],G.in[2*i]);
                connect<> (plin[2*i+1].out[0],G.in[2*i+1]);
                connect<> (G.out[2*i],plout[2*i].in[0]);
                connect<> (G.out[2*i+1],plout[2*i+1].in[0]);

            }
        }
    };
