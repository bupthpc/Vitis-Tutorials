#!/bin/bash
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
export XILINX_XRT=/usr

./host.exe a.xclbin 1000000 &> out.txt
