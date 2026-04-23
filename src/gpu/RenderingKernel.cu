#include "gpu/GPUKernels.cuh"
#include "gpu/GPUUtils.cuh"
#include "gpu/GPUConstants.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

__global__ void GPU::Kernels::computeZBufferPacked64(
	const float* __restrict__ d_X, const float* __restrict__ d_Y, const float* __restrict__ d_Z,
	uint64_t* __restrict__ d_zBuffer,
	int imgWidth, int imgHeight,
	int mlaWidth, int mlaHeight,
	size_t totalPoints
) {
	const int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pointIdx >= totalPoints) return;

	// Load Point Data
	const float pointZ = __ldg(&d_Z[pointIdx]);
	const float pointX = __ldg(&d_X[pointIdx]);
	const float pointY = __ldg(&d_Y[pointIdx]);

	// Cache constants
	const float projectionFactor = GPU::CONSTANTS::d_DISPLAY_DISTANCE_mm * __frcp_rn(pointZ);
	const float fovRadius_mm = fabsf(pointZ * GPU::CONSTANTS::d_TAN_FOV);
	const float mlaPitch_mm = GPU::CONSTANTS::d_MLA_PITCH_mm;
	const float invPixelSize = GPU::CONSTANTS::d_INV_PIXELSIZE_px;
	const float invMlaPitch = GPU::CONSTANTS::d_INV_MLA_PITCH_mm;
	const int halfMicroimageSize = GPU::CONSTANTS::d_HALF_MICROIMAGE_SIZE_px;

	// Find Microlens Bounding Box
	int mlMinI = __float2int_rd((pointX - fovRadius_mm) * invMlaPitch);
	int mlMinJ = __float2int_rd((pointY - fovRadius_mm) * invMlaPitch);
	int mlMaxI = __float2int_ru((pointX + fovRadius_mm) * invMlaPitch);
	int mlMaxJ = __float2int_ru((pointY + fovRadius_mm) * invMlaPitch);

	// Early exit if point is completely outside microlens array
	if (mlMaxI < 0 || mlMinI >= mlaWidth ||
		mlMaxJ < 0 || mlMinJ >= mlaHeight) {
		return;
	}

	// Clamp to valid microlens range
	mlMinI = max(0, mlMinI);
	mlMinJ = max(0, mlMinJ);
	mlMaxI = min(mlaWidth - 1, mlMaxI);
	mlMaxJ = min(mlaHeight - 1, mlMaxJ);

	// Precompute FOV radius squared for comparisons
	const float fovRadiusSq = fovRadius_mm * fovRadius_mm;

	// Flatten microlens loops
	int mlCountI = mlMaxI - mlMinI + 1;
	int mlCountJ = mlMaxJ - mlMinJ + 1;
	int totalMLs = mlCountI * mlCountJ;

	const float startMLCenterX = (mlMinI + 0.5f) * mlaPitch_mm;
	const float startMLCenterY = (mlMinJ + 0.5f) * mlaPitch_mm;

	for (int mlIdx = 0; mlIdx < totalMLs; ++mlIdx) {
		// Convert linear index back to 2D microlens coordinates
		int relI = mlIdx % mlCountI;
		int relJ = mlIdx / mlCountI;

		const float mlCenterX = startMLCenterX + relI * mlaPitch_mm;
		const float mlCenterY = startMLCenterY + relJ * mlaPitch_mm;

		// Relative coordinates
		const float relativeX = pointX - mlCenterX;
		const float relativeY = pointY - mlCenterY;

		// Early rejection: outside FOV
		if (relativeX * relativeX > fovRadiusSq || relativeY * relativeY > fovRadiusSq)
			continue;

		// Projected coordinates
		const float projectedX_mm = __fmaf_rn(relativeX, projectionFactor, mlCenterX);
		const float projectedY_mm = __fmaf_rn(relativeY, projectionFactor, mlCenterY);
		const int projectedX_px = __float2int_rn(projectedX_mm * invPixelSize);
		const int projectedY_px = __float2int_rn(projectedY_mm * invPixelSize);

		// Microimage bounds
		const int mlCenterX_px = __float2int_rn(mlCenterX * invPixelSize);
		const int mlCenterY_px = __float2int_rn(mlCenterY * invPixelSize);
		const int microimageMinX = mlCenterX_px - halfMicroimageSize;
		const int microimageMaxX = mlCenterX_px + halfMicroimageSize;
		const int microimageMinY = mlCenterY_px - halfMicroimageSize;
		const int microimageMaxY = mlCenterY_px + halfMicroimageSize;

		// Early rejection: projected pixel outside microimage
		// Bounds check with single branch
		const bool outOfBounds = (projectedX_px < microimageMinX) || (projectedX_px > microimageMaxX) ||
								 (projectedY_px < microimageMinY) || (projectedY_px > microimageMaxY) ||
								 (projectedX_px < 0) || (projectedX_px >= imgWidth) ||
								 (projectedY_px < 0) || (projectedY_px >= imgHeight);

		if (outOfBounds) continue;

		// Atomic write
		const int pixelIdx = projectedY_px * imgWidth + projectedX_px;

		// Precompute depth key once
		const uint32_t depthKey = GPU::Utils::float_to_ordered_uint(pointZ);
		const uint64_t packedDepthPoint = (uint64_t(depthKey) << 32) | uint64_t(pointIdx);
		atomicMin(&d_zBuffer[pixelIdx], packedDepthPoint);
	}
}
