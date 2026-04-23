#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#include <cuda_runtime.h>

namespace GPU {
	namespace Kernels {
		__global__ void computeZBufferPacked64(
			const float* __restrict__ d_X, const float* __restrict__ d_Y, const float* __restrict__ d_Z,
			uint64_t* __restrict__ d_zBuffer,
			int imgWidth, int imgHeight,
			int mlaWidth, int mlaHeight,
			size_t totalPoints);
	}
}

