#include "gpu/GPUConstants.cuh"
#include "gpu/GPUMemory.cuh"

namespace GPU {
	namespace CONSTANTS {
		__constant__ float d_INV_MLA_PITCH_mm;
		__constant__ float d_INV_PIXELSIZE_px;
		__constant__ int   d_HALF_MICROIMAGE_SIZE_px;
		__constant__ float d_TAN_FOV;
		__constant__ float d_MLA_PITCH_mm;
		__constant__ float d_DISPLAY_DISTANCE_mm;
	}
}

bool GPU::CONSTANTS::initConstants(const SystemSpec& spec, std::string& error) {
	error.clear();

	// Compute constants
	const float inv_pixelSize_px = 1.f / spec.display.pixelSize_mm;
	const float inv_mla_pitch_mm = 1.0f / spec.mla.pitch_mm;
	const float securityMargin = 1.5f;
	const int halfMicroimageSize_px = static_cast<int>(std::round(spec.mla.pitch_mm * 0.5f * inv_pixelSize_px));
	const float tanFOV = spec.mla.pitch_mm  / (2.0f *spec.mla.displayDistance_mm) * securityMargin;

	// Send constants to GPU
	if (!GPU::Memory::sendToConstantMem(inv_pixelSize_px, d_INV_PIXELSIZE_px, error)) return false;
	if (!GPU::Memory::sendToConstantMem(inv_mla_pitch_mm, d_INV_MLA_PITCH_mm, error)) return false;
	if (!GPU::Memory::sendToConstantMem(halfMicroimageSize_px, d_HALF_MICROIMAGE_SIZE_px, error)) return false;
	if (!GPU::Memory::sendToConstantMem(tanFOV, d_TAN_FOV, error)) return false;
	if (!GPU::Memory::sendToConstantMem(spec.mla.pitch_mm, d_MLA_PITCH_mm, error)) return false;
	if (!GPU::Memory::sendToConstantMem(spec.mla.displayDistance_mm, d_DISPLAY_DISTANCE_mm, error)) return false;

	return true;
}

