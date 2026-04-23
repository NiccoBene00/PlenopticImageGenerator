// GPUConstants.cuh
#pragma once
#include <cuda_runtime.h>

#include "data/SystemSpec.hpp"

namespace GPU {
    namespace CONSTANTS {
        extern __constant__ float d_INV_MLA_PITCH_mm;
        extern __constant__ float d_INV_PIXELSIZE_px;
        extern __constant__ int   d_HALF_MICROIMAGE_SIZE_px;
        extern __constant__ float d_TAN_FOV;
        extern __constant__ float d_MLA_PITCH_mm;
        extern __constant__ float d_DISPLAY_DISTANCE_mm;

        bool initConstants(const SystemSpec& spec, std::string& error);
    }
}
