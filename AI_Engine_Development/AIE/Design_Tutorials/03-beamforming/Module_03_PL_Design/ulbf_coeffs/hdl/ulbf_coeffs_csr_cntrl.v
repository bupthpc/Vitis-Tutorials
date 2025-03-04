
/*
Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
*/


`timescale 1ns / 1ps


module ulbf_coeffs_csr_cntrl (
        input wire  [19:0]  BRAM_PORTA_addr,
        input wire          BRAM_PORTA_clk,
        input wire  [31:0]  BRAM_PORTA_din,
        input wire          BRAM_PORTA_en,
        input wire          BRAM_PORTA_we,

        input wire          m0_done,
        input wire          m1_done,
        input wire          m2_done,
        input wire          m3_done,
        input wire          m4_done,
        input wire          m5_done,
        input wire          m6_done,
        input wire          m7_done,
        input wire  [15:0]  addrb_wire0,
        input wire  [15:0]  addrb_wire1,
        input wire  [15:0]  addrb_wire2,
        input wire  [15:0]  addrb_wire3,
        input wire  [15:0]  addrb_wire4,
        input wire  [15:0]  addrb_wire5,
        input wire  [15:0]  addrb_wire6,
        input wire  [15:0]  addrb_wire7,

        
        output wire         go,
        output wire         m_axis_rst,
        output wire [11:0]  block_size,
        output wire [11:0]  niter,
        output wire [15:0]  rollover_addr,

        output reg [31:0] csr_rddata

    );
    
    reg [31:0] ctrl0 = 32'd0;   //reset and go
    reg [31:0] ctrl1 = 32'd256; // block size
    reg [31:0] ctrl2 = 32'd4;  // niter
    reg [31:0] ctrl3 = 32'd1024; //rollover_addr
   

    wire        is_csr, is_write, is_read;
    wire [7:0]  csr_addr;
    wire [31:0] status0;
    wire [31:0] status1;
    wire [31:0] status2;
    wire [31:0] status3;
    wire [31:0] status4;
    wire [31:0] status5;
    wire [31:0] status6;
    wire [31:0] status7;
    wire [31:0] status8;
    //------------------------------------------------------------------
    // control register mapping:
    // 0x0008_0000 : rd - ID 32'h0123_4567
    // 0x0008_0004 : wr - bit 0 - reset
    //               wr - bit 4 - go
    // 0x0008_0008 : wr - 32b of block size. tlast will be asserted for every block_size # cycles
    // 0x0008_000c : wr - niter
    // 0x0008_0010 : wr - rollover_addr
    // 0x0008_0020 : rd - bit 0 - master 0 done
    // 0x0008_0024 : rd - bit 15:0 - master 0 ram row num
    //------------------------------------------------------------------
    


    assign is_csr      = BRAM_PORTA_addr[19];
    assign is_write    = BRAM_PORTA_en && BRAM_PORTA_we;
    assign is_read     = BRAM_PORTA_en && (~BRAM_PORTA_we);
    
    assign csr_addr = BRAM_PORTA_addr[7:0];
    
    always@(posedge BRAM_PORTA_clk)
    begin
          if (is_write && is_csr)
            begin
              case (csr_addr)
                'h4: begin
                    ctrl0 <= BRAM_PORTA_din;
                end
                'h8: begin
                    ctrl1 <= BRAM_PORTA_din;
                end
                'hC: begin
                    ctrl2 <= BRAM_PORTA_din;
                end
                'h10: begin
                    ctrl3 <= BRAM_PORTA_din;
                end
                default: begin
                    ctrl0 <= ctrl0;
                    ctrl1 <= ctrl1;
                    ctrl2 <= ctrl2;
                    ctrl3 <= ctrl3;
                end
              endcase
          end
    end 
    
    
    always@(*)
    begin
      case (csr_addr)
        'h0:  csr_rddata = 32'h0123_4567;
        'h4:  csr_rddata = ctrl0;
        'h8:  csr_rddata = ctrl1;
        'hC:  csr_rddata = ctrl2;
        'h10: csr_rddata = ctrl3;
        'h20: csr_rddata = status0;
        'h24: csr_rddata = status1;
        'h28: csr_rddata = status2;
        'h2C: csr_rddata = status3;
        'h30: csr_rddata = status4;
        'h34: csr_rddata = status5;
        'h38: csr_rddata = status6;
        'h3C: csr_rddata = status7;
        'h40: csr_rddata = status8;
       default: begin
          csr_rddata = 32'd0;
        end
      endcase
    end
   
    assign status0          = {24'd0, m0_done, m1_done, m2_done, m3_done, m4_done, m5_done, m6_done, m7_done};
    assign status1          = {16'd0, addrb_wire0};
    assign status2          = {16'd0, addrb_wire1};
    assign status3          = {16'd0, addrb_wire2};
    assign status4          = {16'd0, addrb_wire3};
    assign status5          = {16'd0, addrb_wire4};
    assign status6          = {16'd0, addrb_wire5};
    assign status7          = {16'd0, addrb_wire6};
    assign status8          = {16'd0, addrb_wire7};
    assign m_axis_rst       = ctrl0[0];
    assign go               = ctrl0[4];
    assign block_size       = ctrl1[11:0];
    assign niter            = ctrl2[11:0];
    assign rollover_addr    = ctrl3[15:0];

endmodule
