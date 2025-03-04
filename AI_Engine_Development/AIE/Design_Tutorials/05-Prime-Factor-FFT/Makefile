#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Author: Mark Rollins

ECHO                               = @echo
export TARGET                      ?= hw_emu

RELEASE=2024.2

TOP_DIR                   ?= $(shell readlink -f .)
PLATFORM_NAME              = xilinx_vck190_base_202420_1
PLATFORM_PATH              = ${PLATFORM_REPO_PATHS}

export PLATFORM            = ${PLATFORM_PATH}/${PLATFORM_NAME}/${PLATFORM_NAME}.xpfm
export SYSROOT             = ${COMMON_IMAGE_VERSAL}/sysroots/cortexa72-cortexa53-xilinx-linux
export KERNEL_IMAGE        = ${COMMON_IMAGE_VERSAL}/Image
export ROOTFS              = ${COMMON_IMAGE_VERSAL}/rootfs.ext4
export PREBUILT_LINUX_PATH = ${COMMON_IMAGE_VERSAL}

##############
# Build Flow #
##############
.PHONY: clean help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all"
	$(ECHO) "      Command to generate everything for this design"
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove all the generated files."

all:	pfa1008

pfa1008:
	$(MAKE) all -C aie/pfa1008
	$(MAKE) all -C hls/pfa1008_dma_src
	$(MAKE) all -C hls/pfa1008_permute_i
	$(MAKE) all -C hls/pfa1008_transpose1
	$(MAKE) all -C hls/pfa1008_transpose2
	$(MAKE) all -C hls/pfa1008_permute_o
	$(MAKE) all -C hls/pfa1008_dma_snk
	$(MAKE) all -C ps_apps/hw_emu
	$(MAKE) all -C vitis

clean:
	$(MAKE) clean -C aie/pfa1008
	$(MAKE) clean -C hls/pfa1008_dma_src
	$(MAKE) clean -C hls/pfa1008_permute_i
	$(MAKE) clean -C hls/pfa1008_transpose1
	$(MAKE) clean -C hls/pfa1008_transpose2
	$(MAKE) clean -C hls/pfa1008_permute_o
	$(MAKE) clean -C hls/pfa1008_dma_snk
	$(MAKE) clean -C ps_apps/hw_emu
	$(MAKE) clean -C vitis
	@/bin/rm -rf .crashReporter*


