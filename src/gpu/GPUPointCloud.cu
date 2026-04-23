#include "gpu/GPUPointCloud.cuh"

#include <cuda_runtime.h>

bool GPU::DevicePointCloud::sendToGPU(const PointCloud& pointCloud, std::string& error)
{
	const int totalPoints = pointCloud.size();

	// Extract color channels
	ArrayU8 red = extractChannel(pointCloud.colors, 0);
	ArrayU8 green = extractChannel(pointCloud.colors, 1);
	ArrayU8 blue = extractChannel(pointCloud.colors, 2);

	// Send data to GPU
	if (!GPU::Memory::sendToGlobalMem(d_X, pointCloud.X.data(), totalPoints, error)) return false;
	if (!GPU::Memory::sendToGlobalMem(d_Y, pointCloud.Y.data(), totalPoints, error)) { return false; }
	if (!GPU::Memory::sendToGlobalMem(d_Z, pointCloud.Z.data(), totalPoints, error)) { return false; }
	if (!GPU::Memory::sendToGlobalMem(d_red, red.data(), totalPoints, error)) { return false; }
	if (!GPU::Memory::sendToGlobalMem(d_green, green.data(), totalPoints, error)) { return false; }
	if (!GPU::Memory::sendToGlobalMem(d_blue, blue.data(), totalPoints, error)) { return false; }

	return true;
}
