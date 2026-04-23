#include "gpu/GPUUtils.cuh"

__global__ void GPU::Utils::resolveZBufferPacked64(
	const uint64_t* __restrict__ d_zBuffer,
	const uint8_t* __restrict__ d_red, const uint8_t* __restrict__ d_green, const uint8_t* __restrict__ d_blue,
	uint8_t* __restrict__ d_output,
	int totalPixels, int totalPoints
) {
	int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixelIdx >= totalPixels) return;

	uint64_t packed = d_zBuffer[pixelIdx];
	// No point hit this pixel 
	if (packed == 0xFFFFFFFFFFFFFFFFull) {
		// Check if already valid (alpha)
		if (d_output[4 * pixelIdx + 3] != 255) {
			d_output[4 * pixelIdx + 0] = 0;
			d_output[4 * pixelIdx + 1] = 0;
			d_output[4 * pixelIdx + 2] = 0;
			d_output[4 * pixelIdx + 3] = 0;
		}
		return;
	}

	// From here on, packed IS guaranteed valid
	uint32_t pointIdx = uint32_t(packed);

	// (Optional but defensive)
	if (pointIdx >= totalPoints)
		return;

	d_output[4 * pixelIdx + 0] = d_red[pointIdx];
	d_output[4 * pixelIdx + 1] = d_green[pointIdx];
	d_output[4 * pixelIdx + 2] = d_blue[pointIdx];
	d_output[4 * pixelIdx + 3] = 255;
}
