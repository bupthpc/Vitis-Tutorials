#!/bin/bash
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Author: Mark Rollins

notice="Vitis_Tutorial_MNIST_ConvNet_on_AIE_ML_notice_container.txt"
zenity --text-info --width 900 --height 1200 --filename=$notice --checkbox="I read and accept the terms"

if [ $? = 0 ]; then
    python -m venv my-venv
    source my-venv/bin/activate
    pip install --upgrade pip
    pip install jupyter
    pip install tensorflow
    pip install matplotlib
    pip install pydot
    pip install bfloat16
    pip list
fi






