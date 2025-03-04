/*
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <cstring>
#include "data.h"

#include "experimental/xrt_kernel.h"
#include "experimental/xrt_graph.h"
#include "experimental/xrt_ip.h"
#include "xrt/xrt_aie.h"
#include "../aie/graph.cpp"

#include "xrt.h"
#include "adf/adf_api/XRTConfig.h"

#define SAMPLES 128
#define NIterations 7

int main(int argc, char* argv[])
{
    	char* xclbinFile=argv[1];
	auto device = xrt::device(0);
    	if(device == nullptr)
		throw std::runtime_error("No valid device handle found. Make sure using right xclOpen index.");
    	auto xclbin_uuid = device.load_xclbin(xclbinFile);
	auto dhdl = xrtDeviceOpenFromXcl(device);

    	int bytes_per_iteration = SAMPLES * 4;//int32 datatype => 4 bytes 
	int size_in_bytes = NIterations * bytes_per_iteration;
	int OUTPUT_SAMPLES_SIZE = size_in_bytes/(4 * 2); //no.of lines of 2 int32 samples of data - w.r.t PL
	//4-bytes for 32-bit and (*2) for 64-bit,(*4) for 128-bit- - So, total (OUTPUT_SIZE) no.of 64-bit samples is available at PLIO
	
	int sizeIn = SAMPLES * NIterations;
	int sizeOut = SAMPLES * NIterations;
	
	auto mm2s_khdl = xrt::kernel(device, xclbin_uuid, "mm2s");
	auto s2mm_1_khdl = xrt::kernel(device, xclbin_uuid, "s2mm:{s2mm_1}");
	auto s2mm_2_khdl = xrt::kernel(device, xclbin_uuid, "s2mm:{s2mm_2}"); //s2mm and mm2s are kernel names
	
	
	auto in_bohdl = xrt::bo(device, sizeIn * sizeof(int), mm2s_khdl.group_id(0));
	auto in_bomapped = in_bohdl.map<int32_t*>();
	memcpy(in_bomapped, Input_x, sizeIn * sizeof(int));//Host only takes 32-bit input. so (7-iterations * 128 samples * 4bytes(int32)) bytes of memory is required. 
							   //Feed 7*128 samples 
	printf("Input memory virtual addr 0x%px\n", in_bomapped);

	in_bohdl.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	
	auto out1_bohdl = xrt::bo(device, sizeOut * sizeof(int), s2mm_1_khdl.group_id(0)); 
	auto out1_bomapped = out1_bohdl.map<int32_t*>();
	memset(out1_bomapped, 0xABCDEF00, sizeOut * sizeof(int));
	printf("Output memory virtual addr 0x%px\n", out1_bomapped);
	
	auto out2_bohdl = xrt::bo(device, sizeOut * sizeof(float),s2mm_2_khdl.group_id(0)); 
	auto out2_bomapped = out2_bohdl.map<float*>();
	memset(out2_bomapped, 0xABCDEF00, sizeOut * sizeof(float));
	printf("Output memory virtual addr 0x%px\n", out2_bomapped);
	
	
	auto mm2s_rhdl = mm2s_khdl(in_bohdl, nullptr, OUTPUT_SAMPLES_SIZE);
									                                                                              
	printf("run mm2s\n");
	
	
	auto s2mm_1_rhdl = s2mm_1_khdl(out1_bohdl, nullptr, OUTPUT_SAMPLES_SIZE);
	auto s2mm_2_rhdl = s2mm_2_khdl(out2_bohdl, nullptr, OUTPUT_SAMPLES_SIZE);
	printf("run s2mm\n");

	//////////////////////////////////////////
	// graph execution for AIE
	//////////////////////////////////////////	
	
	//auto cghdl = xrt::graph(device,xclbin_uuid,"mygraph");
	adf::registerXRT(dhdl, xclbin_uuid.get());
	std::cout<<"Register XRT"<<std::endl;
	
	//Event API's to count the number of samples received from the AI Engine port prior to a graph stall
	event::handle handle_SampleCount = event::start_profiling(mygraph.out0, event::io_stream_running_event_count);
	if(handle_SampleCount==event::invalid_handle){
    		printf("ERROR:Invalid handle. Only two performance counter in a AIE-PL interface tile\n");
    	return 1;
}
	
	printf("graph run\n");
	mygraph.run(NIterations);

	//stop-profiling	

	mygraph.wait();
	long long cycle_count = event::read_profiling(handle_SampleCount);
	printf("Total number of Samples = %d\n", cycle_count);
	event::stop_profiling(handle_SampleCount);//Performance counter is released and cleared
	

	mygraph.end();
	printf("graph end\n");	
	
	
	//////////////////////////////////////////
	// wait for mm2s done
	//////////////////////////////////////////	
	
	mm2s_rhdl.wait();
	printf("After MM2S wait\n");	

	//////////////////////////////////////////
	// wait for s2mm done
	//////////////////////////////////////////	
	
	s2mm_1_rhdl.wait();
	printf("After S2MM_1 wait\n");	
	s2mm_2_rhdl.wait();
	printf("After S2MM_2 wait\n");	

	out1_bohdl.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
	out2_bohdl.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
	
	//////////////////////////////////////////
	// Comparing the execution data to the golden data
	//////////////////////////////////////////	

	int errorCount1 = 0;
	int errorCount2 = 0;
	{
		for (int i = 0; i < OUTPUT_SAMPLES_SIZE; i++)
		{
				if (out1_bomapped[i] != golden_out_shuffle[i])
				{
					printf("Error found - shuffle @ %d, %d != %d\n", i, out1_bomapped[i], golden_out_shuffle[i]);
					errorCount1++;
				}

				if (out2_bomapped[i] != golden_out_upscale[i])
				{
					printf("Error found - upscale @ %d, %f != %f\n", i, out2_bomapped[i], golden_out_upscale[i]);
					errorCount2++;
				}
				
		}

		if (errorCount1 | errorCount2)
			printf("Test failed with %d errors\n", errorCount1);
		else
			printf("TEST PASSED\n");
	}
	
	return (errorCount1 & errorCount2);
}
