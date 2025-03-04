
/*
Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
*/


`timescale 1ns / 1ps


module ulbf_data #(
        parameter TDATA_WIDTH=64,
        parameter TKEEP_WIDTH=TDATA_WIDTH/8,
        parameter MEM_INIT_FILE0="ulbf_din0_hex.mem",
        parameter MEM_INIT_FILE1="ulbf_din1_hex.mem",
        parameter MEM_INIT_FILE2="ulbf_din2_hex.mem",
        parameter MEM_INIT_FILE3="ulbf_din3_hex.mem",
        parameter MEM_INIT_FILE4="ulbf_din4_hex.mem",
        parameter MEM_INIT_FILE5="ulbf_din5_hex.mem",
        parameter MEM_INIT_FILE6="ulbf_din6_hex.mem",
        parameter MEM_INIT_FILE7="ulbf_din7_hex.mem",
        parameter RAM_DEPTH=1536
        ) (

        input wire [22:0]S_AXI_araddr,
        input wire [2:0]S_AXI_arprot,
        output wire S_AXI_arready,
        input wire S_AXI_arvalid,
        input wire [22:0]S_AXI_awaddr,
        input wire [2:0]S_AXI_awprot,
        output wire S_AXI_awready,
        input wire S_AXI_awvalid,
        input wire S_AXI_bready,
        output wire [1:0]S_AXI_bresp,
        output wire S_AXI_bvalid,
        output wire [31:0]S_AXI_rdata,
        input wire S_AXI_rready,
        output wire [1:0]S_AXI_rresp,
        output wire S_AXI_rvalid,
        input wire [31:0]S_AXI_wdata,
        output wire S_AXI_wready,
        input wire [3:0]S_AXI_wstrb,
        input wire S_AXI_wvalid,
        input wire s_axi_aclk,
        input wire s_axi_aresetn,

        input wire m_axis_clk,  
        (* X_INTERFACE_PARAMETER = "CLK_DOMAIN clk_domain_0,LAYERED_METADATA  undef, PHASE 0,FREQ_HZ 400000000,HAS_TLAST 1,HAS_TKEEP 1,HAS_TSTRB 0,HAS_TREADY 1,TUSER_WIDTH 0,TID_WIDTH 0,TDEST_WIDTH 0" *)
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M00_AXIS TVALID" *) output wire                    m00_axis_tvalid,
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M00_AXIS TLAST" *)  output wire                    m00_axis_tlast,  
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M00_AXIS TREADY" *) input  wire                    m00_axis_tready, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M00_AXIS TDATA" *)  output wire [TDATA_WIDTH-1:0]  m00_axis_tdata, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M00_AXIS TKEEP" *)  output wire [TKEEP_WIDTH-1:0]  m00_axis_tkeep,

        (* X_INTERFACE_PARAMETER = "CLK_DOMAIN clk_domain_0,LAYERED_METADATA  undef, PHASE 0,FREQ_HZ 400000000,HAS_TLAST 1,HAS_TKEEP 1,HAS_TSTRB 0,HAS_TREADY 1,TUSER_WIDTH 0,TID_WIDTH 0,TDEST_WIDTH 0" *)
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M01_AXIS TVALID" *) output wire                    m01_axis_tvalid,
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M01_AXIS TLAST" *)  output wire                    m01_axis_tlast,  
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M01_AXIS TREADY" *) input  wire                    m01_axis_tready, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M01_AXIS TDATA" *)  output wire [TDATA_WIDTH-1:0]  m01_axis_tdata, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M01_AXIS TKEEP" *)  output wire [TKEEP_WIDTH-1:0]  m01_axis_tkeep,

        (* X_INTERFACE_PARAMETER = "CLK_DOMAIN clk_domain_0,LAYERED_METADATA  undef, PHASE 0,FREQ_HZ 400000000,HAS_TLAST 1,HAS_TKEEP 1,HAS_TSTRB 0,HAS_TREADY 1,TUSER_WIDTH 0,TID_WIDTH 0,TDEST_WIDTH 0" *)
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M02_AXIS TVALID" *) output wire                    m02_axis_tvalid,
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M02_AXIS TLAST" *)  output wire                    m02_axis_tlast,  
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M02_AXIS TREADY" *) input  wire                    m02_axis_tready, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M02_AXIS TDATA" *)  output wire [TDATA_WIDTH-1:0]  m02_axis_tdata, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M02_AXIS TKEEP" *)  output wire [TKEEP_WIDTH-1:0]  m02_axis_tkeep,

        (* X_INTERFACE_PARAMETER = "CLK_DOMAIN clk_domain_0,LAYERED_METADATA  undef, PHASE 0,FREQ_HZ 400000000,HAS_TLAST 1,HAS_TKEEP 1,HAS_TSTRB 0,HAS_TREADY 1,TUSER_WIDTH 0,TID_WIDTH 0,TDEST_WIDTH 0" *)
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M03_AXIS TVALID" *) output wire                    m03_axis_tvalid,
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M03_AXIS TLAST" *)  output wire                    m03_axis_tlast,  
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M03_AXIS TREADY" *) input  wire                    m03_axis_tready, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M03_AXIS TDATA" *)  output wire [TDATA_WIDTH-1:0]  m03_axis_tdata, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M03_AXIS TKEEP" *)  output wire [TKEEP_WIDTH-1:0]  m03_axis_tkeep, 
 
        (* X_INTERFACE_PARAMETER = "CLK_DOMAIN clk_domain_0,LAYERED_METADATA  undef, PHASE 0,FREQ_HZ 400000000,HAS_TLAST 1,HAS_TKEEP 1,HAS_TSTRB 0,HAS_TREADY 1,TUSER_WIDTH 0,TID_WIDTH 0,TDEST_WIDTH 0" *)
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M04_AXIS TVALID" *) output wire                    m04_axis_tvalid,
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M04_AXIS TLAST" *)  output wire                    m04_axis_tlast,  
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M04_AXIS TREADY" *) input  wire                    m04_axis_tready, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M04_AXIS TDATA" *)  output wire [TDATA_WIDTH-1:0]  m04_axis_tdata, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M04_AXIS TKEEP" *)  output wire [TKEEP_WIDTH-1:0]  m04_axis_tkeep,

        (* X_INTERFACE_PARAMETER = "CLK_DOMAIN clk_domain_0,LAYERED_METADATA  undef, PHASE 0,FREQ_HZ 400000000,HAS_TLAST 1,HAS_TKEEP 1,HAS_TSTRB 0,HAS_TREADY 1,TUSER_WIDTH 0,TID_WIDTH 0,TDEST_WIDTH 0" *)
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M05_AXIS TVALID" *) output wire                    m05_axis_tvalid,
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M05_AXIS TLAST" *)  output wire                    m05_axis_tlast,  
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M05_AXIS TREADY" *) input  wire                    m05_axis_tready, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M05_AXIS TDATA" *)  output wire [TDATA_WIDTH-1:0]  m05_axis_tdata, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M05_AXIS TKEEP" *)  output wire [TKEEP_WIDTH-1:0]  m05_axis_tkeep,

        (* X_INTERFACE_PARAMETER = "CLK_DOMAIN clk_domain_0,LAYERED_METADATA  undef, PHASE 0,FREQ_HZ 400000000,HAS_TLAST 1,HAS_TKEEP 1,HAS_TSTRB 0,HAS_TREADY 1,TUSER_WIDTH 0,TID_WIDTH 0,TDEST_WIDTH 0" *)
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M06_AXIS TVALID" *) output wire                    m06_axis_tvalid,
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M06_AXIS TLAST" *)  output wire                    m06_axis_tlast,  
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M06_AXIS TREADY" *) input  wire                    m06_axis_tready, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M06_AXIS TDATA" *)  output wire [TDATA_WIDTH-1:0]  m06_axis_tdata, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M06_AXIS TKEEP" *)  output wire [TKEEP_WIDTH-1:0]  m06_axis_tkeep,

        (* X_INTERFACE_PARAMETER = "CLK_DOMAIN clk_domain_0,LAYERED_METADATA  undef, PHASE 0,FREQ_HZ 400000000,HAS_TLAST 1,HAS_TKEEP 1,HAS_TSTRB 0,HAS_TREADY 1,TUSER_WIDTH 0,TID_WIDTH 0,TDEST_WIDTH 0" *)
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M07_AXIS TVALID" *) output wire                    m07_axis_tvalid,
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M07_AXIS TLAST" *)  output wire                    m07_axis_tlast,  
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M07_AXIS TREADY" *) input  wire                    m07_axis_tready, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M07_AXIS TDATA" *)  output wire [TDATA_WIDTH-1:0]  m07_axis_tdata, 
        (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M07_AXIS TKEEP" *)  output wire [TKEEP_WIDTH-1:0]  m07_axis_tkeep
        
        
    );
	
	wire [22:0] bram_addr_a;
	wire        bram_clk_a;
	wire        bram_en_a;
	reg [31:0] bram_rddata_a;
	wire        bram_rst_a;
	wire [3:0]  bram_we_a;
	wire [31:0] bram_wrdata_a;

	reg [19:0] bram_addr_a0, bram_addr_a1, bram_addr_a2, bram_addr_a3;
	reg [19:0] bram_addr_a4, bram_addr_a5, bram_addr_a6, bram_addr_a7;
	reg [31:0] bram_wrdata_a0, bram_wrdata_a1, bram_wrdata_a2, bram_wrdata_a3;
	reg [31:0] bram_wrdata_a4, bram_wrdata_a5, bram_wrdata_a6, bram_wrdata_a7;
	wire [31:0] bram_rddata_a0, bram_rddata_a1, bram_rddata_a2, bram_rddata_a3;
	wire [31:0] bram_rddata_a4, bram_rddata_a5, bram_rddata_a6, bram_rddata_a7;
	reg	   bram_en_a0, bram_en_a1, bram_en_a2, bram_en_a3;
	reg	   bram_en_a4, bram_en_a5, bram_en_a6, bram_en_a7;
	reg [3:0]  bram_we_a0, bram_we_a1, bram_we_a2, bram_we_a3;
	reg [3:0]  bram_we_a4, bram_we_a5, bram_we_a6, bram_we_a7;

    wire        go_bram, go_maxis;
    wire        m_axis_rst_bram, m_axis_rst_maxis;
    wire [11:0] block_size_bram, block_size_maxis;
    wire [11:0] niter_bram, niter_maxis;
    wire [15:0] rollover_addr_bram, rollover_addr_maxis;
    wire        m0_done_bram, m0_done_maxis;
    wire        m1_done_bram, m1_done_maxis;
    wire        m2_done_bram, m2_done_maxis;
    wire        m3_done_bram, m3_done_maxis;
    wire        m4_done_bram, m4_done_maxis;
    wire        m5_done_bram, m5_done_maxis;
    wire        m6_done_bram, m6_done_maxis;
    wire        m7_done_bram, m7_done_maxis;
    wire [15:0] addrb_wire_bram0, addrb_wire_maxis0;
    wire [15:0] addrb_wire_bram1, addrb_wire_maxis1;
    wire [15:0] addrb_wire_bram2, addrb_wire_maxis2;
    wire [15:0] addrb_wire_bram3, addrb_wire_maxis3;
    wire [15:0] addrb_wire_bram4, addrb_wire_maxis4;
    wire [15:0] addrb_wire_bram5, addrb_wire_maxis5;
    wire [15:0] addrb_wire_bram6, addrb_wire_maxis6;
    wire [15:0] addrb_wire_bram7, addrb_wire_maxis7;

    wire [31:0] csr_rddata;

	
    	design_ulbf_data_wrapper  design_ulbf_data_wrapper_i (
            .S_AXI_araddr(S_AXI_araddr),
            .S_AXI_arprot(S_AXI_arprot),
            .S_AXI_arready(S_AXI_arready),
            .S_AXI_arvalid(S_AXI_arvalid),
            .S_AXI_awaddr(S_AXI_awaddr),
            .S_AXI_awprot(S_AXI_awprot),
            .S_AXI_awready(S_AXI_awready),
            .S_AXI_awvalid(S_AXI_awvalid),
            .S_AXI_bready(S_AXI_bready),
            .S_AXI_bresp(S_AXI_bresp),
            .S_AXI_bvalid(S_AXI_bvalid),
            .S_AXI_rdata(S_AXI_rdata),
            .S_AXI_rready(S_AXI_rready),
            .S_AXI_rresp(S_AXI_rresp),
            .S_AXI_rvalid(S_AXI_rvalid),
            .S_AXI_wdata(S_AXI_wdata),
            .S_AXI_wready(S_AXI_wready),
            .S_AXI_wstrb(S_AXI_wstrb),
            .S_AXI_wvalid(S_AXI_wvalid),
            .s_axi_aclk(s_axi_aclk),
            .s_axi_aresetn(s_axi_aresetn),

            .bram_addr_a(bram_addr_a),
            .bram_clk_a(bram_clk_a),
            .bram_en_a(bram_en_a),
            .bram_rddata_a(bram_rddata_a),
            .bram_rst_a(bram_rst_a),
            .bram_we_a(bram_we_a),
            .bram_wrdata_a(bram_wrdata_a)

        );

            
	always@(*)
	begin
		case (bram_addr_a[22:20])
		3'b000: begin
			bram_addr_a0 = bram_addr_a[19:0];
			bram_en_a0 = bram_en_a;
			bram_we_a0 = bram_we_a;
			bram_wrdata_a0 = bram_wrdata_a;
			bram_addr_a1 = 19'b0;
			bram_en_a1 = 1'b0;
			bram_we_a1 = 4'd0;
			bram_wrdata_a1 = 32'd0;
			bram_addr_a2 = 19'b0;
			bram_en_a2 = 1'b0;
			bram_we_a2 = 4'd0;
			bram_wrdata_a2 = 32'd0;
			bram_addr_a3 = 19'b0;
			bram_en_a3 = 1'b0;
			bram_we_a3 = 4'd0;
			bram_wrdata_a3 = 32'd0;
			bram_addr_a4 = 19'b0;
			bram_en_a4 = 1'b0;
			bram_we_a4 = 4'd0;
			bram_wrdata_a4 = 32'd0;
			bram_addr_a5 = 19'b0;
			bram_en_a5 = 1'b0;
			bram_we_a5 = 4'd0;
			bram_wrdata_a5 = 32'd0;
			bram_addr_a6 = 19'b0;
			bram_en_a6 = 1'b0;
			bram_we_a6 = 4'd0;
			bram_wrdata_a6 = 32'd0;
			bram_addr_a7 = 19'b0;
			bram_en_a7 = 1'b0;
			bram_we_a7 = 4'd0;
			bram_wrdata_a7 = 32'd0;
			bram_rddata_a = bram_rddata_a0;
			end
		3'b001: begin
			bram_addr_a0 = 19'b0;
			bram_en_a0 = 1'b0;
			bram_we_a0 = 4'd0;
			bram_wrdata_a0 = 32'd0;
			bram_addr_a1 = bram_addr_a[19:0];
			bram_en_a1 = bram_en_a;
			bram_we_a1 = bram_we_a;
			bram_wrdata_a1 = bram_wrdata_a;
			bram_addr_a2 = 19'b0;
			bram_en_a2 = 1'b0;
			bram_we_a2 = 4'd0;
			bram_wrdata_a2 = 32'd0;
			bram_addr_a3 = 19'b0;
			bram_en_a3 = 1'b0;
			bram_we_a3 = 4'd0;
			bram_wrdata_a3 = 32'd0;
			bram_addr_a4 = 19'b0;
			bram_en_a4 = 1'b0;
			bram_we_a4 = 4'd0;
			bram_wrdata_a4 = 32'd0;
			bram_addr_a5 = 19'b0;
			bram_en_a5 = 1'b0;
			bram_we_a5 = 4'd0;
			bram_wrdata_a5 = 32'd0;
			bram_addr_a6 = 19'b0;
			bram_en_a6 = 1'b0;
			bram_we_a6 = 4'd0;
			bram_wrdata_a6 = 32'd0;
			bram_addr_a7 = 19'b0;
			bram_en_a7 = 1'b0;
			bram_we_a7 = 4'd0;
			bram_wrdata_a7 = 32'd0;
			bram_rddata_a = bram_rddata_a1;
			end
		3'b010: begin
			bram_addr_a0 = 19'b0;
			bram_en_a0 = 1'b0;
			bram_we_a0 = 4'd0;
			bram_wrdata_a0 = 32'd0;
			bram_addr_a1 = 19'b0;
			bram_en_a1 = 1'b0;
			bram_we_a1 = 4'd0;
			bram_wrdata_a1 = 32'd0;
			bram_addr_a2 = bram_addr_a[19:0];
			bram_en_a2 = bram_en_a;
			bram_we_a2 = bram_we_a;
			bram_wrdata_a2 = bram_wrdata_a;
			bram_addr_a3 = 19'b0;
			bram_en_a3 = 1'b0;
			bram_we_a3 = 4'd0;
			bram_wrdata_a3 = 32'd0;
			bram_addr_a4 = 19'b0;
			bram_en_a4 = 1'b0;
			bram_we_a4 = 4'd0;
			bram_wrdata_a4 = 32'd0;
			bram_addr_a5 = 19'b0;
			bram_en_a5 = 1'b0;
			bram_we_a5 = 4'd0;
			bram_wrdata_a5 = 32'd0;
			bram_addr_a6 = 19'b0;
			bram_en_a6 = 1'b0;
			bram_we_a6 = 4'd0;
			bram_wrdata_a6 = 32'd0;
			bram_addr_a7 = 19'b0;
			bram_en_a7 = 1'b0;
			bram_we_a7 = 4'd0;
			bram_wrdata_a7 = 32'd0;
			bram_rddata_a = bram_rddata_a2;
			end
		3'b011: begin
			bram_addr_a0 = 19'b0;
			bram_en_a0 = 1'b0;
			bram_we_a0 = 4'd0;
			bram_wrdata_a0 = 32'd0;
			bram_addr_a1 = 19'b0;
			bram_en_a1 = 1'b0;
			bram_we_a1 = 4'd0;
			bram_wrdata_a1 = 32'd0;
			bram_addr_a2 = 19'b0;
			bram_en_a2 = 1'b0;
			bram_we_a2 = 4'd0;
			bram_wrdata_a2 = 32'd0;
			bram_addr_a3 = bram_addr_a[19:0];
			bram_en_a3 = bram_en_a;
			bram_we_a3 = bram_we_a;
			bram_wrdata_a3 = bram_wrdata_a;
			bram_addr_a4 = 19'b0;
			bram_en_a4 = 1'b0;
			bram_we_a4 = 4'd0;
			bram_wrdata_a4 = 32'd0;
			bram_addr_a5 = 19'b0;
			bram_en_a5 = 1'b0;
			bram_we_a5 = 4'd0;
			bram_wrdata_a5 = 32'd0;
			bram_addr_a6 = 19'b0;
			bram_en_a6 = 1'b0;
			bram_we_a6 = 4'd0;
			bram_wrdata_a6 = 32'd0;
			bram_addr_a7 = 19'b0;
			bram_en_a7 = 1'b0;
			bram_we_a7 = 4'd0;
			bram_wrdata_a7 = 32'd0;
			bram_rddata_a = bram_rddata_a3;
            end
		3'b100: begin
			bram_addr_a0 = 19'b0;
			bram_en_a0 = 1'b0;
			bram_we_a0 = 4'd0;
			bram_wrdata_a0 = 32'd0;
			bram_addr_a1 = 19'b0;
			bram_en_a1 = 1'b0;
			bram_we_a1 = 4'd0;
			bram_wrdata_a1 = 32'd0;
			bram_addr_a2 = 19'b0;
			bram_en_a2 = 1'b0;
			bram_we_a2 = 4'd0;
			bram_wrdata_a2 = 32'd0;
			bram_addr_a3 = 19'b0;
			bram_en_a3 = 1'b0;
			bram_we_a3 = 4'b0;
			bram_wrdata_a3 = 32'b0;
			bram_addr_a4 = bram_addr_a[19:0];
			bram_en_a4 = bram_en_a;
			bram_we_a4 = bram_we_a;
			bram_wrdata_a4 = bram_wrdata_a;
			bram_addr_a5 = 19'b0;
			bram_en_a5 = 1'b0;
			bram_we_a5 = 4'd0;
			bram_wrdata_a5 = 32'd0;
			bram_addr_a6 = 19'b0;
			bram_en_a6 = 1'b0;
			bram_we_a6 = 4'd0;
			bram_wrdata_a6 = 32'd0;
			bram_addr_a7 = 19'b0;
			bram_en_a7 = 1'b0;
			bram_we_a7 = 4'd0;
			bram_wrdata_a7 = 32'd0;
			bram_rddata_a = bram_rddata_a4;
			end
		3'b101: begin
			bram_addr_a0 = 19'b0;
			bram_en_a0 = 1'b0;
			bram_we_a0 = 4'd0;
			bram_wrdata_a0 = 32'd0;
			bram_addr_a1 = 19'b0;
			bram_en_a1 = 1'b0;
			bram_we_a1 = 4'd0;
			bram_wrdata_a1 = 32'd0;
			bram_addr_a2 = 19'b0;
			bram_en_a2 = 1'b0;
			bram_we_a2 = 4'd0;
			bram_wrdata_a2 = 32'd0;
			bram_addr_a3 = 19'b0;
			bram_en_a3 = 1'b0;
			bram_we_a3 = 4'b0;
			bram_wrdata_a3 = 32'b0;
            bram_addr_a4 = 19'b0;
			bram_en_a4 = 1'b0;
			bram_we_a4 = 4'd0;
			bram_wrdata_a4 = 32'd0;
			bram_addr_a5 = bram_addr_a[19:0];
			bram_en_a5 = bram_en_a;
			bram_we_a5 = bram_we_a;
			bram_wrdata_a5 = bram_wrdata_a;
			bram_addr_a6 = 19'b0;
			bram_en_a6 = 1'b0;
			bram_we_a6 = 4'd0;
			bram_wrdata_a6 = 32'd0;
			bram_addr_a7 = 19'b0;
			bram_en_a7 = 1'b0;
			bram_we_a7 = 4'd0;
			bram_wrdata_a7 = 32'd0;
			bram_rddata_a = bram_rddata_a5;
			end
		3'b110: begin
			bram_addr_a0 = 19'b0;
			bram_en_a0 = 1'b0;
			bram_we_a0 = 4'd0;
			bram_wrdata_a0 = 32'd0;
			bram_addr_a1 = 19'b0;
			bram_en_a1 = 1'b0;
			bram_we_a1 = 4'd0;
			bram_wrdata_a1 = 32'd0;
			bram_addr_a2 = 19'b0;
			bram_en_a2 = 1'b0;
			bram_we_a2 = 4'd0;
			bram_wrdata_a2 = 32'd0;
			bram_addr_a3 = 19'b0;
			bram_en_a3 = 1'b0;
			bram_we_a3 = 4'b0;
			bram_wrdata_a3 = 32'b0;
            bram_addr_a4 = 19'b0;
			bram_en_a4 = 1'b0;
			bram_we_a4 = 4'd0;
			bram_wrdata_a4 = 32'd0;
			bram_addr_a5 = 19'b0;
			bram_en_a5 = 1'b0;
			bram_we_a5 = 4'd0;
			bram_wrdata_a5 = 32'd0;
			bram_addr_a6 = bram_addr_a[19:0];
			bram_en_a6 = bram_en_a;
			bram_we_a6 = bram_we_a;
			bram_wrdata_a6 = bram_wrdata_a;
			bram_addr_a7 = 19'b0;
			bram_en_a7 = 1'b0;
			bram_we_a7 = 4'd0;
			bram_wrdata_a7 = 32'd0;
			bram_rddata_a = bram_rddata_a6;
			end

		3'b111: begin
			bram_addr_a0 = 19'b0;
			bram_en_a0 = 1'b0;
			bram_we_a0 = 4'd0;
			bram_wrdata_a0 = 32'd0;
			bram_addr_a1 = 19'b0;
			bram_en_a1 = 1'b0;
			bram_we_a1 = 4'd0;
			bram_wrdata_a1 = 32'd0;
			bram_addr_a2 = 19'b0;
			bram_en_a2 = 1'b0;
			bram_we_a2 = 4'd0;
			bram_wrdata_a2 = 32'd0;
			bram_addr_a3 = 19'b0;
			bram_en_a3 = 1'b0;
			bram_we_a3 = 4'b0;
			bram_wrdata_a3 = 32'b0;
            bram_addr_a4 = 19'b0;
			bram_en_a4 = 1'b0;
			bram_we_a4 = 4'd0;
			bram_wrdata_a4 = 32'd0;
			bram_addr_a5 = 19'b0;
			bram_en_a5 = 1'b0;
			bram_we_a5 = 4'd0;
			bram_wrdata_a5 = 32'd0;
			bram_addr_a6 = 19'b0;
			bram_en_a6 = 1'b0;
			bram_we_a6 = 4'd0;
			bram_wrdata_a6 = 32'd0;
			bram_addr_a7 = bram_addr_a[19:0];
			bram_en_a7 = bram_en_a;
			bram_we_a7 = bram_we_a;
			bram_wrdata_a7 = bram_wrdata_a;
			bram_rddata_a = bram_rddata_a7;
			end
		default: begin
			bram_addr_a0 = 19'b0;
			bram_en_a0 = 1'b0;
			bram_we_a0 = 4'd0;
			bram_wrdata_a0 = 32'd0;
			bram_addr_a1 = 19'b0;
			bram_en_a1 = 1'b0;
			bram_we_a1 = 4'd0;
			bram_wrdata_a1 = 32'd0;
			bram_addr_a2 = 19'b0;
			bram_en_a2 = 1'b0;
			bram_we_a2 = 4'd0;
			bram_wrdata_a2 = 32'd0;
			bram_addr_a3 = 19'b0;
			bram_en_a3 = 1'b0;
			bram_we_a3 = 4'd0;
			bram_wrdata_a3 = 32'd0;
			bram_addr_a4 = 19'b0;
			bram_en_a4 = 1'b0;
			bram_we_a4 = 4'd0;
			bram_wrdata_a4 = 32'd0;
			bram_addr_a5 = 19'b0;
			bram_en_a5 = 1'b0;
			bram_we_a5 = 4'd0;
			bram_wrdata_a5 = 32'd0;
			bram_addr_a6 = 19'b0;
			bram_en_a6 = 1'b0;
			bram_we_a6 = 4'd0;
			bram_wrdata_a6 = 32'd0;
			bram_addr_a7 = 19'b0;
			bram_en_a7 = 1'b0;
			bram_we_a7 = 4'd0;
			bram_wrdata_a7 = 32'd0;
			bram_rddata_a = 32'd0;
			end
		endcase
	end
            
	ulbf_data_master #(
		.TDATA_WIDTH(TDATA_WIDTH),
		.TKEEP_WIDTH(TDATA_WIDTH/8),
		.MEM_INIT_FILE(MEM_INIT_FILE0),
		.RAM_DEPTH(RAM_DEPTH)
		) master_i_0 (
		.bram_clk_a(bram_clk_a),
		.bram_rst_a(bram_rst_a),
		.bram_we_a(bram_we_a0),
		.bram_addr_a(bram_addr_a0),
		.bram_en_a(bram_en_a0),
		.bram_rddata_a(bram_rddata_a0),
		.bram_wrdata_a(bram_wrdata_a0),

		.m_axis_clk(m_axis_clk),  
		.m00_axis_tvalid(m00_axis_tvalid),
		.m00_axis_tlast(m00_axis_tlast),  
		.m00_axis_tready(m00_axis_tready), 
		.m00_axis_tdata(m00_axis_tdata), 
		.m00_axis_tkeep(m00_axis_tkeep),

        .csr_rddata(csr_rddata),
        .go_maxis(go_maxis),
        .m_axis_rst_maxis(m_axis_rst_maxis),
        .niter_maxis(niter_maxis),
        .block_size_maxis(block_size_maxis),
        .rollover_addr_maxis(rollover_addr_maxis),

        .m0_done_maxis(m0_done_maxis),
       .addrb_wire_maxis(addrb_wire_maxis0)
	); 

	ulbf_data_master #(
		.TDATA_WIDTH(TDATA_WIDTH),
		.TKEEP_WIDTH(TDATA_WIDTH/8),
		.MEM_INIT_FILE(MEM_INIT_FILE1),
		.RAM_DEPTH(RAM_DEPTH)
		) master_i_1 (
		.bram_clk_a(bram_clk_a),
		.bram_rst_a(bram_rst_a),
		.bram_we_a(bram_we_a1),
		.bram_addr_a(bram_addr_a1),
		.bram_en_a(bram_en_a1),
		.bram_rddata_a(bram_rddata_a1),
		.bram_wrdata_a(bram_wrdata_a1),

		.m_axis_clk(m_axis_clk),  
		.m00_axis_tvalid(m01_axis_tvalid),
		.m00_axis_tlast(m01_axis_tlast),  
		.m00_axis_tready(m01_axis_tready), 
		.m00_axis_tdata(m01_axis_tdata), 
		.m00_axis_tkeep(m01_axis_tkeep),

        .csr_rddata(csr_rddata),
        .go_maxis(go_maxis),
        .m_axis_rst_maxis(m_axis_rst_maxis),
        .niter_maxis(niter_maxis),
        .block_size_maxis(block_size_maxis),
        .rollover_addr_maxis(rollover_addr_maxis),

        .m0_done_maxis(m1_done_maxis),
        .addrb_wire_maxis(addrb_wire_maxis1)
	); 

	ulbf_data_master #(
		.TDATA_WIDTH(TDATA_WIDTH),
		.TKEEP_WIDTH(TDATA_WIDTH/8),
		.MEM_INIT_FILE(MEM_INIT_FILE2),
		.RAM_DEPTH(RAM_DEPTH)
		) master_i_2 (
		.bram_clk_a(bram_clk_a),
		.bram_rst_a(bram_rst_a),
		.bram_we_a(bram_we_a2),
		.bram_addr_a(bram_addr_a2),
		.bram_en_a(bram_en_a2),
		.bram_rddata_a(bram_rddata_a2),
		.bram_wrdata_a(bram_wrdata_a2),

		.m_axis_clk(m_axis_clk),  
		.m00_axis_tvalid(m02_axis_tvalid),
		.m00_axis_tlast(m02_axis_tlast),  
		.m00_axis_tready(m02_axis_tready), 
		.m00_axis_tdata(m02_axis_tdata), 
		.m00_axis_tkeep(m02_axis_tkeep),

        .csr_rddata(csr_rddata),
        .go_maxis(go_maxis),
        .m_axis_rst_maxis(m_axis_rst_maxis),
        .niter_maxis(niter_maxis),
        .block_size_maxis(block_size_maxis),
        .rollover_addr_maxis(rollover_addr_maxis),

        .m0_done_maxis(m2_done_maxis),
        .addrb_wire_maxis(addrb_wire_maxis2)
	); 

	ulbf_data_master #(
		.TDATA_WIDTH(TDATA_WIDTH),
		.TKEEP_WIDTH(TDATA_WIDTH/8),
		.MEM_INIT_FILE(MEM_INIT_FILE3),
		.RAM_DEPTH(RAM_DEPTH)
		) master_i_3 (
		.bram_clk_a(bram_clk_a),
		.bram_rst_a(bram_rst_a),
		.bram_we_a(bram_we_a3),
		.bram_addr_a(bram_addr_a3),
		.bram_en_a(bram_en_a3),
		.bram_rddata_a(bram_rddata_a3),
		.bram_wrdata_a(bram_wrdata_a3),

		.m_axis_clk(m_axis_clk),  
		.m00_axis_tvalid(m03_axis_tvalid),
		.m00_axis_tlast(m03_axis_tlast),  
		.m00_axis_tready(m03_axis_tready), 
		.m00_axis_tdata(m03_axis_tdata), 
		.m00_axis_tkeep(m03_axis_tkeep),

        .csr_rddata(csr_rddata),
        .go_maxis(go_maxis),
        .m_axis_rst_maxis(m_axis_rst_maxis),
        .niter_maxis(niter_maxis),
        .block_size_maxis(block_size_maxis),
        .rollover_addr_maxis(rollover_addr_maxis),

        .m0_done_maxis(m3_done_maxis),
        .addrb_wire_maxis(addrb_wire_maxis3)
	); 

            
	ulbf_data_master #(
		.TDATA_WIDTH(TDATA_WIDTH),
		.TKEEP_WIDTH(TDATA_WIDTH/8),
		.MEM_INIT_FILE(MEM_INIT_FILE4),
		.RAM_DEPTH(RAM_DEPTH)
		) master_i_4 (
		.bram_clk_a(bram_clk_a),
		.bram_rst_a(bram_rst_a),
		.bram_we_a(bram_we_a4),
		.bram_addr_a(bram_addr_a4),
		.bram_en_a(bram_en_a4),
		.bram_rddata_a(bram_rddata_a4),
		.bram_wrdata_a(bram_wrdata_a4),

		.m_axis_clk(m_axis_clk),  
		.m00_axis_tvalid(m04_axis_tvalid),
		.m00_axis_tlast(m04_axis_tlast),  
		.m00_axis_tready(m04_axis_tready), 
		.m00_axis_tdata(m04_axis_tdata), 
		.m00_axis_tkeep(m04_axis_tkeep),

        .csr_rddata(csr_rddata),
        .go_maxis(go_maxis),
        .m_axis_rst_maxis(m_axis_rst_maxis),
        .niter_maxis(niter_maxis),
        .block_size_maxis(block_size_maxis),
        .rollover_addr_maxis(rollover_addr_maxis),

        .m0_done_maxis(m4_done_maxis),
       .addrb_wire_maxis(addrb_wire_maxis4)
	); 

	ulbf_data_master #(
		.TDATA_WIDTH(TDATA_WIDTH),
		.TKEEP_WIDTH(TDATA_WIDTH/8),
		.MEM_INIT_FILE(MEM_INIT_FILE5),
		.RAM_DEPTH(RAM_DEPTH)
		) master_i_5 (
		.bram_clk_a(bram_clk_a),
		.bram_rst_a(bram_rst_a),
		.bram_we_a(bram_we_a5),
		.bram_addr_a(bram_addr_a5),
		.bram_en_a(bram_en_a5),
		.bram_rddata_a(bram_rddata_a5),
		.bram_wrdata_a(bram_wrdata_a5),

		.m_axis_clk(m_axis_clk),  
		.m00_axis_tvalid(m05_axis_tvalid),
		.m00_axis_tlast(m05_axis_tlast),  
		.m00_axis_tready(m05_axis_tready), 
		.m00_axis_tdata(m05_axis_tdata), 
		.m00_axis_tkeep(m05_axis_tkeep),

        .csr_rddata(csr_rddata),
        .go_maxis(go_maxis),
        .m_axis_rst_maxis(m_axis_rst_maxis),
        .niter_maxis(niter_maxis),
        .block_size_maxis(block_size_maxis),
        .rollover_addr_maxis(rollover_addr_maxis),

        .m0_done_maxis(m5_done_maxis),
        .addrb_wire_maxis(addrb_wire_maxis5)
	); 

	ulbf_data_master #(
		.TDATA_WIDTH(TDATA_WIDTH),
		.TKEEP_WIDTH(TDATA_WIDTH/8),
		.MEM_INIT_FILE(MEM_INIT_FILE6),
		.RAM_DEPTH(RAM_DEPTH)
		) master_i_6 (
		.bram_clk_a(bram_clk_a),
		.bram_rst_a(bram_rst_a),
		.bram_we_a(bram_we_a6),
		.bram_addr_a(bram_addr_a6),
		.bram_en_a(bram_en_a6),
		.bram_rddata_a(bram_rddata_a6),
		.bram_wrdata_a(bram_wrdata_a6),

		.m_axis_clk(m_axis_clk),  
		.m00_axis_tvalid(m06_axis_tvalid),
		.m00_axis_tlast(m06_axis_tlast),  
		.m00_axis_tready(m06_axis_tready), 
		.m00_axis_tdata(m06_axis_tdata), 
		.m00_axis_tkeep(m06_axis_tkeep),

        .csr_rddata(csr_rddata),
        .go_maxis(go_maxis),
        .m_axis_rst_maxis(m_axis_rst_maxis),
        .niter_maxis(niter_maxis),
        .block_size_maxis(block_size_maxis),
        .rollover_addr_maxis(rollover_addr_maxis),

        .m0_done_maxis(m6_done_maxis),
        .addrb_wire_maxis(addrb_wire_maxis6)
	); 

	ulbf_data_master #(
		.TDATA_WIDTH(TDATA_WIDTH),
		.TKEEP_WIDTH(TDATA_WIDTH/8),
		.MEM_INIT_FILE(MEM_INIT_FILE7),
		.RAM_DEPTH(RAM_DEPTH)
		) master_i_7 (
		.bram_clk_a(bram_clk_a),
		.bram_rst_a(bram_rst_a),
		.bram_we_a(bram_we_a7),
		.bram_addr_a(bram_addr_a7),
		.bram_en_a(bram_en_a7),
		.bram_rddata_a(bram_rddata_a7),
		.bram_wrdata_a(bram_wrdata_a7),

		.m_axis_clk(m_axis_clk),  
		.m00_axis_tvalid(m07_axis_tvalid),
		.m00_axis_tlast(m07_axis_tlast),  
		.m00_axis_tready(m07_axis_tready), 
		.m00_axis_tdata(m07_axis_tdata), 
		.m00_axis_tkeep(m07_axis_tkeep),

        .csr_rddata(csr_rddata),
        .go_maxis(go_maxis),
        .m_axis_rst_maxis(m_axis_rst_maxis),
        .niter_maxis(niter_maxis),
        .block_size_maxis(block_size_maxis),
        .rollover_addr_maxis(rollover_addr_maxis),

        .m0_done_maxis(m7_done_maxis),
        .addrb_wire_maxis(addrb_wire_maxis7)
	); 

    ulbf_data_csr_cntrl  ulbf_data_csr(
        .BRAM_PORTA_addr(bram_addr_a[19:0]),
        .BRAM_PORTA_clk(bram_clk_a),
        .BRAM_PORTA_din(bram_wrdata_a),
        .BRAM_PORTA_en(bram_en_a),
        .BRAM_PORTA_we(bram_we_a),

        .m0_done(m0_done_bram),
        .m1_done(m1_done_bram),
        .m2_done(m2_done_bram),
        .m3_done(m3_done_bram),
        .m4_done(m4_done_bram),
        .m5_done(m5_done_bram),
        .m6_done(m6_done_bram),
        .m7_done(m7_done_bram),
        .addrb_wire0(addrb_wire_bram0),
        .addrb_wire1(addrb_wire_bram1),
        .addrb_wire2(addrb_wire_bram2),
        .addrb_wire3(addrb_wire_bram3),
        .addrb_wire4(addrb_wire_bram4),
        .addrb_wire5(addrb_wire_bram5),
        .addrb_wire6(addrb_wire_bram6),
        .addrb_wire7(addrb_wire_bram7),
        
        .go(go_bram),
        .m_axis_rst(m_axis_rst_bram),
        .block_size(block_size_bram),
        .niter(niter_bram),
        .rollover_addr(rollover_addr_bram),

        .csr_rddata(csr_rddata)
    );


    ulbf_data_cdc     plio_master_64b_cdc_i(
        .m_axis_clk(m_axis_clk),
        .bram_clk_a(bram_clk_a),

        .go_bram(go_bram),
        .m_axis_rst_bram(m_axis_rst_bram),
        .m0_done_maxis(m0_done_maxis),
        .m1_done_maxis(m1_done_maxis),
        .m2_done_maxis(m2_done_maxis),
        .m3_done_maxis(m3_done_maxis),
        .m4_done_maxis(m4_done_maxis),
        .m5_done_maxis(m5_done_maxis),
        .m6_done_maxis(m6_done_maxis),
        .m7_done_maxis(m7_done_maxis),
        .niter_bram(niter_bram),
        .block_size_bram(block_size_bram),
        .rollover_addr_bram(rollover_addr_bram),
        .addrb_wire_maxis0(addrb_wire_maxis0),
        .addrb_wire_maxis1(addrb_wire_maxis1),
        .addrb_wire_maxis2(addrb_wire_maxis2),
        .addrb_wire_maxis3(addrb_wire_maxis3),
        .addrb_wire_maxis4(addrb_wire_maxis4),
        .addrb_wire_maxis5(addrb_wire_maxis5),
        .addrb_wire_maxis6(addrb_wire_maxis6),
        .addrb_wire_maxis7(addrb_wire_maxis7),

        .go_maxis(go_maxis),
        .m_axis_rst_maxis(m_axis_rst_maxis),
        .m0_done_bram(m0_done_bram),
        .m1_done_bram(m1_done_bram),
        .m2_done_bram(m2_done_bram),
        .m3_done_bram(m3_done_bram),
        .m4_done_bram(m4_done_bram),
        .m5_done_bram(m5_done_bram),
        .m6_done_bram(m6_done_bram),
        .m7_done_bram(m7_done_bram),
        .niter_maxis(niter_maxis),
        .block_size_maxis(block_size_maxis),
        .rollover_addr_maxis(rollover_addr_maxis),
        .addrb_wire_bram0(addrb_wire_bram0),
        .addrb_wire_bram1(addrb_wire_bram1),
        .addrb_wire_bram2(addrb_wire_bram2),
        .addrb_wire_bram3(addrb_wire_bram3),
        .addrb_wire_bram4(addrb_wire_bram4),
        .addrb_wire_bram5(addrb_wire_bram5),
        .addrb_wire_bram6(addrb_wire_bram6),
        .addrb_wire_bram7(addrb_wire_bram7)
	);

endmodule

