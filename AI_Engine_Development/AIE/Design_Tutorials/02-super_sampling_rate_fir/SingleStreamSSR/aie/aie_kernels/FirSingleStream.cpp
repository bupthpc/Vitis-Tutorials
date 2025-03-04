/*
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
*/



 #include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "FirSingleStream.h"

template <int NSamples,int ShiftAcc>
void SingleStream::FIR_MultiKernel_cout<NSamples,ShiftAcc>::filter(input_stream<cint16>* sin,output_cascade<cacc48>* cout)
{
	v8cint16 taps =  *(v8cint16*) weights;
	v16cint16 *ptr_delay_line = (v16cint16 *)delay_line;
	v16cint16 data = *ptr_delay_line;

	v4cacc48 acc = undef_v4cacc48();



// Computes 16 samples per iteration
	for(int i=0;i<NSamples/16;i++)
		chess_prepare_for_pipelining
		chess_loop_range(NSamples/16,NSamples/16)
	{
		acc = mul4(data,1,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,3,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 2, readincr_v<4>(sin));
		acc = mac4(acc,data,5,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,7,0x3210,1,taps,6,0x0000,1);
		writeincr(cout,acc);

		acc = mul4(data,5,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,7,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 3, readincr_v<4>(sin));
		acc = mac4(acc,data,9,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,11,0x3210,1,taps,6,0x0000,1);
		writeincr(cout,acc);

		acc = mul4(data,9,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,11,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 0, readincr_v<4>(sin));
		acc = mac4(acc,data,13,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,15,0x3210,1,taps,6,0x0000,1);
		writeincr(cout,acc);

		acc = mul4(data,13,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,15,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 1, readincr_v<4>(sin));
		acc = mac4(acc,data,1,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,3,0x3210,1,taps,6,0x0000,1);
		writeincr(cout,acc);

	}

	*ptr_delay_line = data;
}




template <int NSamples,int ShiftAcc>
void SingleStream::FIR_MultiKernel_cincout<NSamples,ShiftAcc>::filter(input_stream<cint16>* sin,input_cascade<cacc48>* cin,output_cascade<cacc48>* cout)
{

	v8cint16 taps =  *(v8cint16*) weights;
	v16cint16 *ptr_delay_line = (v16cint16 *)delay_line;
	v16cint16 data = *ptr_delay_line;

	v4cacc48 acc = undef_v4cacc48();


// Computes 16 samples per iteration
	for(int i=0;i<NSamples/16;i++)
		chess_prepare_for_pipelining
		chess_loop_range(NSamples/16,NSamples/16)
	{
		acc = readincr_v<4>(cin);
		acc = mac4(acc,data,1,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,3,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 2, readincr_v<4>(sin));
		acc = mac4(acc,data,5,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,7,0x3210,1,taps,6,0x0000,1);
		writeincr(cout,acc);

		acc = readincr_v<4>(cin);
		acc = mac4(acc,data,5,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,7,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 3, readincr_v<4>(sin));
		acc = mac4(acc,data,9,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,11,0x3210,1,taps,6,0x0000,1);
		writeincr(cout,acc);

		acc = readincr_v<4>(cin);
		acc = mac4(acc,data,9,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,11,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 0, readincr_v<4>(sin));
		acc = mac4(acc,data,13,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,15,0x3210,1,taps,6,0x0000,1);
		writeincr(cout,acc);

		acc = readincr_v<4>(cin);
		acc = mac4(acc,data,13,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,15,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 1, readincr_v<4>(sin));
		acc = mac4(acc,data,1,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,3,0x3210,1,taps,6,0x0000,1);
		writeincr(cout,acc);

	}

	*ptr_delay_line = data;
}


template <int NSamples,int ShiftAcc>
void SingleStream::FIR_MultiKernel_cin<NSamples,ShiftAcc>::filter(input_stream<cint16>* sin,input_cascade<cacc48>* cin,output_stream<cint16>* sout)
{

	v8cint16 taps =  *(v8cint16*) weights;
	v16cint16 *ptr_delay_line = (v16cint16 *)delay_line;
	v16cint16 data = *ptr_delay_line;

	v4cacc48 acc = undef_v4cacc48();


// Computes 16 samples per iteration
	for(int i=0;i<NSamples/16;i++)
		chess_prepare_for_pipelining
		chess_loop_range(NSamples/16,NSamples/16)
	{
		acc = readincr_v<4>(cin);
		acc = mac4(acc,data,1,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,3,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 2, readincr_v<4>(sin));
		acc = mac4(acc,data,5,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,7,0x3210,1,taps,6,0x0000,1);
		writeincr(sout,srs(acc,ShiftAcc));

		acc = readincr_v<4>(cin);
		acc = mac4(acc,data,5,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,7,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 3, readincr_v<4>(sin));
		acc = mac4(acc,data,9,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,11,0x3210,1,taps,6,0x0000,1);
		writeincr(sout,srs(acc,ShiftAcc));

		acc = readincr_v<4>(cin);
		acc = mac4(acc,data,9,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,11,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 0, readincr_v<4>(sin));
		acc = mac4(acc,data,13,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,15,0x3210,1,taps,6,0x0000,1);
		writeincr(sout,srs(acc,ShiftAcc));

		acc = readincr_v<4>(cin);
		acc = mac4(acc,data,13,0x3210,1,taps,0,0x0000,1);
		acc = mac4(acc,data,15,0x3210,1,taps,2,0x0000,1);
		data = upd_v(data, 1, readincr_v<4>(sin));
		acc = mac4(acc,data,1,0x3210,1,taps,4,0x0000,1);
		acc = mac4(acc,data,3,0x3210,1,taps,6,0x0000,1);
		writeincr(sout,srs(acc,ShiftAcc));

	}

	*ptr_delay_line = data;
}


void SingleStream::FIRinit(const int Delay)
{
    for (int i = 0; i < Delay; ++i)
    {
        get_ss(0);
    }
}
