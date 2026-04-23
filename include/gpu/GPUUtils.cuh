#pragma once

#include <cstdint>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

namespace GPU {
    namespace Utils {
        inline bool CUDA_CHECK_ERR(cudaError_t err, std::string& errorMsg) {
            if (err != cudaSuccess) {
                errorMsg = std::string("CUDA error: ") + cudaGetErrorString(err);
                std::cerr << errorMsg << std::endl;
                return false;
            }
            return true;
        }

        // Convert float depth to ordered uint for atomicMin z-buffer operations: Handles negative values and NaN by mapping to high uint values
        __device__ inline uint32_t float_to_ordered_uint(float depth)
        {
            uint32_t u = __float_as_uint(depth);
            uint32_t mask = (u & 0x80000000) ? 0xFFFFFFFF : 0x80000000;
            return u ^ mask;
        }

        // Unpack 64-bit Z-Buffer to RGBA
        __global__ void resolveZBufferPacked64(
            const uint64_t* __restrict__ d_zBuffer,
            const uint8_t* __restrict__ d_red, const uint8_t* __restrict__ d_green, const uint8_t* __restrict__ d_blue,
            uint8_t* __restrict__ d_output,
            int totalPixels, int totalPoints);
    }
}