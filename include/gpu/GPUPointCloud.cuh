#pragma once

#include <cstdint>
#include <data/PointCloud.hpp>

#include "gpu/GPUMemory.cuh"

namespace GPU {
	struct DevicePointCloud {
        float* d_X = nullptr;
        float* d_Y = nullptr;
        float* d_Z = nullptr;
        uint8_t* d_red = nullptr;
        uint8_t* d_green = nullptr;
        uint8_t* d_blue = nullptr;

        bool sendToGPU(const PointCloud& pointCloud, std::string& error);

        DevicePointCloud() = default;
        DevicePointCloud(const DevicePointCloud&) = delete;
        DevicePointCloud& operator=(const DevicePointCloud&) = delete;
        ~DevicePointCloud() {
            GPU::Memory::freeDataMem(d_X);
            GPU::Memory::freeDataMem(d_Y);
            GPU::Memory::freeDataMem(d_Z);
            GPU::Memory::freeDataMem(d_red);
            GPU::Memory::freeDataMem(d_green);
            GPU::Memory::freeDataMem(d_blue);
        }

	};
}