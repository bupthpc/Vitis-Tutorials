#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Author: Faisal El-Shabani

DO_COSIM        := false
ECHO            := @echo

KERNEL          := merge_8x4_wrapper
KERNEL_SRC      := merge_8x4.cpp
XO              := ${KERNEL}.xo

.PHONY: clean all help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all"
	$(ECHO) "      Command to generate the kernel used in this design"
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove all the generated files."

all: ${XO}

${XO}: ${KERNEL_SRC}
	v++ --config hls.cfg --compile --mode hls --work_dir ${KERNEL}
ifeq (${DO_COSIM},true)
	vitis-run --mode hls --config hls.cfg --csim --work_dir ${KERNEL}
	vitis-run --mode hls --config hls.cfg --cosim --work_dir ${KERNEL}
	vitis-run --mode hls --config hls.cfg --impl --work_dir ${KERNEL}
endif

clean:
	rm -f $(XO)* *.log vitis-comp.json
	rm -rf .Xil _x/ .crashReporter*
	rm -rf ${KERNEL}
