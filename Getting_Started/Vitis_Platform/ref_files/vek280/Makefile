#/*
#Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT
#*/


ROOT_DIR=$(realpath $(dir $(lastword $(MAKEFILE_LIST))))
ABS_BUILD_PATH=$(ROOT_DIR)/build
ABS_HW_PATH=$(ABS_BUILD_PATH)/vivado
BD_TCL=run.tcl
ABS_BD_TCL=$(ROOT_DIR)/$(BD_TCL)								  
VERSION=2024.2
COMMON_IMAGE_VERSAL ?=${PLATFORM_REPO_PATHS}/sw/versal/xilinx-versal-common-v$(VERSION)/
SYSROOT=$(COMMON_IMAGE_VERSAL)/sysroots/cortexa72-cortexa53-xilinx-linux/
XSA_NAME=vek280_custom
PLATFORM_NAME=vek280_custom
PLATFORM ?= $(ROOT_DIR)/ws/$(PLATFORM_NAME)/export/$(PLATFORM_NAME)/$(PLATFORM_NAME).xpfm
#PLATFORM ?=$(ROOT_DIR)/test/vek280_custom/export/vek280_custom/vek280_custom.xpfm

VADD_WORK=$(ROOT_DIR)/vadd_work
AIE_WORK = $(ROOT_DIR)/aieml_system_design/




.PHONY: help 

help:
	@echo "Makefile usage"
	@echo "make all COMMON_IMAGE_VERSAL=<path/to/common_image/>  	## to generate platform with pre-built software components and run vadd application in software emulation mode"
	@echo "make sd_card COMMON_IMAGE_VERSAL=<path/to/common_image/>  	## to generate platform with pre-built software components and run vadd application on board"

#all: check-common-image hw  dtb  pfm vadd_emu 
sd_card:check-common-image hw    pfm vadd_hw aieml_hw
all: check-common-image hw    pfm vadd_emu aieml_emu
#target : check  whether the common image is ready 
check-common-image:
ifeq ($(wildcard $(COMMON_IMAGE_VERSAL)/Image),)
	@echo "'COMMON_IMAGE_VERSAL' is not accessible, please download common image from Xilinx Downloads website and set this flag to specify the common image path."
	@make help
	@echo "exit ....." && exit 1
endif

# target: hw
# Build  Vivado design and export XSA
hw: $(ABS_BD_TCL)
	mkdir -p $(ABS_HW_PATH)
	cd $(ABS_HW_PATH); vivado -mode batch -notrace -source $(ABS_BD_TCL) -tclargs


# Target: pfm
# Generate the Vitis platform with XSCT
# Input: vck190_custom.xsa in build/vivado directory
pfm: $(ABS_HW_PATH)/$(XSA_NAME).xsa
	@echo "INFO: Creating Platform $(XSA_NAME)"
	mkdir -p tmp && export XILINX_VITIS_DATA_DIR="tmp" && vitis -s platform_creation.py --platform_name $(PLATFORM_NAME) --xsa_path $(ABS_HW_PATH)/$(XSA_NAME).xsa --boot $(COMMON_IMAGE_VERSAL) --user_dtsi $(ROOT_DIR)/system-user.dtsi
	

cpVaddCode:
	mkdir -p vadd_work
	cp $(XILINX_VITIS)/examples/vadd/* vadd_work
	cp makefile_vadd vadd_work/Makefile




vadd_emu: cpVaddCode
	$(MAKE) -C $(VADD_WORK) run TARGET=hw_emu PLATFORM=$(PLATFORM) DEV_ARCH=versal HOST_ARCH=aarch64 EDGE_COMMON_SW=$(COMMON_IMAGE_VERSAL) SYSROOT=$(SYSROOT)/

# vadd_hw is a place-holder for manual build and test on hardware
vadd_hw: cpVaddCode
	$(MAKE) -C $(VADD_WORK) sd_card TARGET=hw PLATFORM=$(PLATFORM) DEV_ARCH=versal HOST_ARCH=aarch64 EDGE_COMMON_SW=$(COMMON_IMAGE_VERSAL) SYSROOT=$(SYSROOT)/
aieml_emu:  $(SYSROOT)/
	cp makefile_aieml $(AIE_WORK)/Makefile
	$(MAKE) -C $(AIE_WORK) run TARGET=hw_emu PLATFORM=$(PLATFORM) DEV_ARCH=versal HOST_ARCH=aarch64 EDGE_COMMON_SW=$(COMMON_IMAGE_VERSAL) SYSROOT=$(SYSROOT)/
aieml_hw:  $(SYSROOT)/
	cp makefile_aieml $(AIE_WORK)/Makefile
	$(MAKE) -C $(AIE_WORK) clean
	$(MAKE) -C $(AIE_WORK) all TARGET=hw PLATFORM=$(PLATFORM) HOST_ARCH=aarch64 DEV_ARCH=versal EDGE_COMMON_SW=$(COMMON_IMAGE_VERSAL) SYSROOT=$(SYSROOT)


clean:
	rm -rf $(DTB_PLT_NAME) $(PLATFORM_NAME) tmp  *journal.py __pycache__ logs ip_cache vitis_journal* 
	-$(MAKE) -C $(VADD_WORK) clean
	-$(MAKE) -C $(AIE_WORK) clean

ultraclean: clean
	-rm -rf $(ABS_BUILD_PATH) device-tree-xlnx/ logs tmp ws
	-rm -rf $(VADD_WORK) 
	make ultraclean  -C  $(AIE_WORK) 