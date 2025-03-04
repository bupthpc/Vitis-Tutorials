
/*
Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
*/

#ifndef __INCLUDE_H__
#define __INCLUDE_H__
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include <aie_api/utils.hpp>

// Downlink Beamforming parameters
#define NUM_PRBS     8
using namespace adf;
using namespace aie;
// In Out Parameters
#define SHFT              17
#define IN_DATA_BLKSZ     (NUM_PRBS*8*12)
#define IN_COEF_BLKSZ     (NUM_PRBS*8*8)
#define OUT_DATA_BLKSZ    (NUM_PRBS*8*12)
#define IN_DATA_WINSZ     (4*IN_DATA_BLKSZ)
#define IN_COEF_WINSZ     (4*IN_COEF_BLKSZ)
#define OUT_DATA_WINSZ    (4*OUT_DATA_BLKSZ)

#define READSCD (getc_scd())
#define READSS0 (getc_wss(0))


#endif /* __INCLUDE_H__ */
