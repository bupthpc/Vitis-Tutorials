#!/bin/sh
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Author: Peifang Zhou, Bachir Berkane, Mark Rollins
#

cp -p ../scanner/data/output_x86sim_0.txt data/input_x86sim_0.txt
cp -p ../scanner/data/output_x86sim_1.txt data/input_x86sim_1.txt
cp -p ../scanner/data/output_x86sim_2.txt data/input_x86sim_2.txt
cp -p ../scanner/data/output_x86sim_3.txt data/input_x86sim_3.txt
cp -p ../scanner/data/output_x86sim_4.txt data/input_x86sim_4.txt
cp -p ../scanner/data/output_x86sim_5.txt data/input_x86sim_5.txt
cp -p ../scanner/data/output_x86sim_6.txt data/input_x86sim_6.txt
cp -p ../scanner/data/output_x86sim_7.txt data/input_x86sim_7.txt
grep -v T ../scanner/data/output_aiesim_0.txt > data/input_aiesim_0.txt
grep -v T ../scanner/data/output_aiesim_1.txt > data/input_aiesim_1.txt
grep -v T ../scanner/data/output_aiesim_2.txt > data/input_aiesim_2.txt
grep -v T ../scanner/data/output_aiesim_3.txt > data/input_aiesim_3.txt
grep -v T ../scanner/data/output_aiesim_4.txt > data/input_aiesim_4.txt
grep -v T ../scanner/data/output_aiesim_5.txt > data/input_aiesim_5.txt
grep -v T ../scanner/data/output_aiesim_6.txt > data/input_aiesim_6.txt
grep -v T ../scanner/data/output_aiesim_7.txt > data/input_aiesim_7.txt

