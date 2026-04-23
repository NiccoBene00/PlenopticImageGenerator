#include "gpu/GPUInterface.cuh"
#include "gpu/GPUPointCloud.cuh"
#include "gpu/GPUContext.cuh"
#include "gpu/GPUKernels.cuh"
#include "gpu/GPUUtils.cuh"
#include "gpu/GPUMemory.cuh"
#include "gpu/GPUConstants.cuh"

#include <opencv2/opencv.hpp>

bool GPU::Render::renderPlenopticImage(cv::Mat& plenopticImage, PointCloud& ptCloud, const SystemSpec& spec, std::string& errorMsg)
{
	errorMsg.clear();

	const size_t imageWidth = plenopticImage.cols;
	const size_t imageHeight = plenopticImage.rows;
	const size_t nChannels = plenopticImage.channels();
	const size_t mlaWidth = spec.mla.countX;
	const size_t mlaHeight = spec.mla.countY;
	const size_t totalPoints = ptCloud.size();
	const size_t totalPixels = imageHeight * imageWidth;
	const size_t imageSize = totalPixels * nChannels;

	// Send Point cloud data
	GPU::DevicePointCloud d_pointCloud;
	if (!d_pointCloud.sendToGPU(ptCloud, errorMsg)) { return false; }

	if (!GPU::CONSTANTS::initConstants(spec, errorMsg)) { return false; };

	// Buffers allocation and initialization out, zbuffer, writecount
	GPU::Context::ContextPacked64 context;
	if (!GPU::Memory::sendToGlobalMem(context.d_output, plenopticImage.data, imageSize, errorMsg)) { return false; }
	if (!GPU::Memory::allocGlobalMem(context.d_zBuffer, totalPixels, errorMsg)) { return false; }
	if (!GPU::Memory::initGlobalMemArray(context.d_zBuffer, 0xFFFFFFFFFFFFFFFFull, totalPixels, errorMsg)) { return false; }

	// Launch kernel
	int nThreads = 128;
	int nBlocksPoints = static_cast<int>((totalPoints + nThreads - 1) / nThreads);
	GPU::Kernels::computeZBufferPacked64 << <nBlocksPoints, nThreads >> > (
		d_pointCloud.d_X, d_pointCloud.d_Y, d_pointCloud.d_Z,
		context.d_zBuffer,
		imageWidth, imageHeight,
		mlaWidth, mlaHeight,
		totalPoints
		);

	nThreads = 256;
	int nBlocksPixels = static_cast<int>((totalPixels + nThreads - 1) / nThreads);
	GPU::Utils::resolveZBufferPacked64 << <nBlocksPixels, nThreads >> > (
		context.d_zBuffer,
		d_pointCloud.d_red, d_pointCloud.d_green, d_pointCloud.d_blue,
		context.d_output,
		totalPixels, totalPoints);

	if (!GPU::Utils::CUDA_CHECK_ERR(cudaDeviceSynchronize(), errorMsg)) { return false; }

	if (!GPU::Memory::getFromGlobalMem(plenopticImage.data, context.d_output, imageSize, errorMsg)) { return false; }

	return true;
}

