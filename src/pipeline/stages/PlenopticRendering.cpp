#include "pipeline/stages/PlenopticRendering.hpp"

#include <iostream>
#include <algorithm>
#include <execution>
#include <vector>

#include "gpu/GPUInterface.cuh"
#include "utils/PrintUtils.hpp"

PointCloud PlenopticRendering::maskSelectZ0Range(const PointCloud& ptCloud, const float Z0Range_mm)
{
	const size_t total = ptCloud.size();

	// Filter out any point outside Z0 range
	Mask Z0Mask = (ptCloud.Z.abs() < Z0Range_mm);
	const int Z0Count = Z0Mask.count();

	// Pre allocating
	PointCloud pcMasked;
	pcMasked.resize(Z0Count);

	// Getting pointers
	const float* xPtr = ptCloud.X.data();
	const float* yPtr = ptCloud.Y.data();
	const float* zPtr = ptCloud.Z.data();
	const uint16_t* pxPtr = ptCloud.px.data();
	const uint16_t* pyPtr = ptCloud.py.data();
	const cv::Vec3b* cPtr = ptCloud.colors.data();

	float* xmPtr = pcMasked.X.data();
	float* ymPtr = pcMasked.Y.data();
	float* zmPtr = pcMasked.Z.data();
	uint16_t* pxmPtr = pcMasked.px.data();
	uint16_t* pymPtr = pcMasked.py.data();
	cv::Vec3b* cmPtr = pcMasked.colors.data();

	// Masking
	Eigen::Index validIdx = 0;
	for (size_t pixelIdx = 0; pixelIdx < total; pixelIdx++) {
		if (!Z0Mask(pixelIdx)) continue;	// Pixels out of range, do nothing

		xmPtr[validIdx] = xPtr[pixelIdx];
		ymPtr[validIdx] = yPtr[pixelIdx];
		zmPtr[validIdx] = zPtr[pixelIdx];
		pxmPtr[validIdx] = pxPtr[pixelIdx];
		pymPtr[validIdx] = pyPtr[pixelIdx];
		cmPtr[validIdx] = cPtr[pixelIdx];

		++validIdx;
	}
	return pcMasked;
}

std::unordered_map<int, std::vector<int>> PlenopticRendering::groupPixelsPerML_linearIdx(const PointCloud& ptCloudAroundZ0, const MLASpec& mla)
{
	const size_t total = ptCloudAroundZ0.size();
	const size_t mlaWidth = mla.countX;
	const size_t mlaHeight = mla.countY;

	std::unordered_map<int, std::vector<int>> groups;
	groups.reserve(mlaWidth * mlaHeight);

#pragma omp parallel for
	for (uint32_t idx = 0; idx < total; idx++) {
		// Find in which ML is the pixel located (integer division)
		const int mi = static_cast<int>(ptCloudAroundZ0.X[idx] / mla.pitch_mm);
		const int mj = static_cast<int>(ptCloudAroundZ0.Y[idx] / mla.pitch_mm);

		// Bounds check
		if (mi < 0 || mi >= mlaWidth || mj < 0 || mj >= mlaHeight)
			continue;

		// Linearize the idx and add to group
		const int mlIdx = mj * mlaWidth + mi;
		groups[mlIdx].push_back(idx);
	}
	return groups;
}

bool PlenopticRendering::initMicroimageZ0(const PointCloud& ptCloud, const Config& config, const SystemSpec& spec, cv::Mat& plenopticImage)
{
	std::cout << "Initiliazing Plane Z0...\n";

	// Mask and retain only points and pixels colors that are with Z0 range
	const float Zthreshold = 2 * spec.mla.displayDistance_mm;
	std::cout << "\t* Z threshold: " << Zthreshold << std::endl;
	PointCloud masked = maskSelectZ0Range(ptCloud, Zthreshold);

	// Group pixels per ML
	std::unordered_map<int, std::vector<int>> groups = groupPixelsPerML_linearIdx(masked, spec.mla);

	// Constants for calculations later
	const size_t mlaWidth = spec.mla.countX;
	const size_t mlaHeight = spec.mla.countY;
	const float mlaPitch_px = spec.mla.pitch_mm / spec.display.pixelSize_mm;
	const int rectSize = static_cast<int>(std::round(mlaPitch_px) + 1);

	// Color the ML with the average color of all pixels on its group
#pragma omp parallel for
	for (const auto& [mlIdx, pxVector] : groups) {
		if (pxVector.empty()) continue;    // if no pixels in this vector

		// Accumulate the color of the pixels for this ML
		cv::Vec3d avgColor(0, 0, 0);
		for (const uint32_t& pxIdx : pxVector) {
			if (pxIdx < 0 || pxIdx >= masked.size())
				continue;
			avgColor += masked.getColor(pxIdx);
		}
		avgColor /= static_cast<double>(pxVector.size());

		// Colorize plenoptic image ML region with average color
		const int mi = mlIdx % mlaWidth;
		const int mj = mlIdx / mlaWidth;
		// Top-left corner
		const int mi_px = static_cast<int>(std::lround(mi * mlaPitch_px));
		const int mj_px = static_cast<int>(std::lround(mj * mlaPitch_px));

		cv::Rect mlColoring(mi_px, mj_px, rectSize, rectSize);
		cv::rectangle(plenopticImage, mlColoring, cv::Scalar(avgColor[0], avgColor[1], avgColor[2], 255), -1);
	}

	return true;
}

bool PlenopticRendering::renderPlenopticImage(const SystemSpec& spec, PointCloud& ptCloud, cv::Mat& plenopticImage)
{
	std::cout << "Rendering Plenoptic Image\n";

	if (!GPU::Render::renderPlenopticImage(plenopticImage, ptCloud, spec, error)) {
		setError(error);
		return false;
	}


	if (plenopticImage.empty()) {
		setError("CUDA plenoptic image result is empty.");
		return false;
	}

	return true;
}

bool PlenopticRendering::setupSteps()
{
	steps.clear();

	registerStep(
		"Initilize Microimage Z0",
		[this](PipelineData& d) {
			return initMicroimageZ0(d.pointCloud, d.config, d.spec, d.plenopticImage);
		},
		true
	);

	registerStep(
		"Render Plenoptic Image",
		[this](PipelineData& d) {
			return renderPlenopticImage(d.spec, d.pointCloud, d.plenopticImage);
		},
		true
	);

	return true;
}


