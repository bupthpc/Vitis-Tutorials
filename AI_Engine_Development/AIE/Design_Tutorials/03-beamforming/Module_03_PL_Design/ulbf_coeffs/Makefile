#Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT



# =========================================================
# PL KERNEL NAME
# =========================================================
KERNEL_NAME := ulbf_coeffs

# =========================================================
# DEVICE SELECTION
# =========================================================
DEVICE_NAME := xcvc1902-vsva2197-2MP-e-S

# =========================================================
# XO CREATION SCRIPT
# =========================================================
CREATE_XO_TCL := run_$(KERNEL_NAME).tcl

# =========================================================
# DEPENDENCIES
# =========================================================
DEPS := $(CREATE_XO_TCL)
DEPS += bd_$(KERNEL_NAME).tcl
DEPS += $(shell find ./hdl -type f -name "*.v")

# =========================================================
# COMMAND OPTIONS
# =========================================================
RUN_OPT := NO_SIM

# =========================================================
# BUILD PRODUCTS
# =========================================================
BUILD_DIR := ../build
KERNEL_XO := $(KERNEL_NAME).xo

kernel: $(BUILD_DIR)/$(KERNEL_NAME).xo
	echo "[Info] All complete - xo $< exists"

$(BUILD_DIR)/$(KERNEL_NAME).xo: $(DEPS)
	vivado -mode batch -source $(CREATE_XO_TCL) -tclargs $(RUN_OPT) $(DEVICE_NAME)

clean:
	rm -rf vivado*.log
	rm -rf vivado*.jou
	rm -rf project_$(KERNEL_NAME) kernel_$(KERNEL_NAME)_auto.xml
	rm -rf .Xil
	rm -rf $(BUILD_DIR)/$(KERNEL_NAME) 
	rm -rf $(BUILD_DIR)/$(KERNEL_NAME).xo
	$(MAKE) clean -C sim
