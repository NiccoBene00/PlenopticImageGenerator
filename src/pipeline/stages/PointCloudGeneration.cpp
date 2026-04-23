#include "pipeline/stages/PointCloudGeneration.hpp"

#include <iostream>
#include <algorithm>
#include <vector>

bool PointCloudGeneration::initPointCloud(const cv::Mat& rgbImage, const cv::Mat& depthMap, PointCloud& ptCloud)
{
	auto t0 = std::chrono::high_resolution_clock::now();//time benchmark

	CV_Assert(!rgbImage.empty() && !depthMap.empty());
	CV_Assert(rgbImage.rows == depthMap.rows && rgbImage.cols == depthMap.cols);
	CV_Assert(rgbImage.type() == CV_8UC3);
	CV_Assert(depthMap.type() == CV_32FC1);

	std::cout << "Point Cloud Initialization:\n\t";

	const size_t width = rgbImage.cols;
	const size_t height = rgbImage.rows;
	const size_t totalPixels = width * height;

	// Resize arrays once
	ptCloud.px.resize(totalPixels);
	ptCloud.py.resize(totalPixels);
	ptCloud.Z.resize(totalPixels);
	ptCloud.colors.resize(totalPixels);

	uint16_t* pxPtr = ptCloud.px.data();
	uint16_t* pyPtr = ptCloud.py.data();
	float* ZPtr = ptCloud.Z.data();
	cv::Vec3b* colPtr = ptCloud.colors.data();

	const cv::Vec3b* imgPtr = rgbImage.ptr<cv::Vec3b>(0);
	const float* depthPtr = depthMap.ptr<float>(0);

	// Initialize pixels coordinates
	Eigen::Map<Eigen::Array<uint16_t, Eigen::Dynamic, 1>> pxMap(pxPtr, totalPixels);
	Eigen::Map<Eigen::Array<uint16_t, Eigen::Dynamic, 1>> pyMap(pyPtr, totalPixels);

	// Precompute a single row of x-coordinates
	Eigen::Array<uint16_t, Eigen::Dynamic, 1> pxRow = Eigen::Array<uint16_t, Eigen::Dynamic, 1>::LinSpaced(width, 0, width - 1);

#pragma omp parallel for schedule(static)
	for (size_t y = 0; y < height; ++y)
	{
		const size_t rowStart = y * width;
		pxMap.segment(rowStart, width) = pxRow;           // SIMD x
		pyMap.segment(rowStart, width).setConstant(y);    // SIMD y
	}

	// Copy depth & colors
	//-------old------//
	std::memcpy(ZPtr, depthPtr, totalPixels * sizeof(float));
	std::memcpy(colPtr, imgPtr, totalPixels * sizeof(cv::Vec3b));
	//----------------//

	//-------new------//
	// Copy depth safely
	/*
	if (depthMap.isContinuous()) {
    	std::memcpy(ZPtr, depthPtr, totalPixels * sizeof(float));
	} else {
    	for (size_t y = 0; y < height; ++y) {
        	const float* rowPtr = depthMap.ptr<float>(y);
        	std::memcpy(ZPtr + y * width, rowPtr, width * sizeof(float));
    	}
	}


	// Copy colors safely
	if (rgbImage.isContinuous()) {
    	std::memcpy(colPtr, imgPtr, totalPixels * sizeof(cv::Vec3b));
	} else {
    	for (size_t y = 0; y < height; ++y) {
        	const cv::Vec3b* rowPtr = rgbImage.ptr<cv::Vec3b>(y);
        	std::memcpy(colPtr + y * width, rowPtr, width * sizeof(cv::Vec3b));
    	}
	}
	*/
	//----------------//

	auto t1 = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	std::cout << "[CPU][TIME] initPointCloud: " << ms << " ms\n";

	return true;
}

bool PointCloudGeneration::project2Dto3D(const DatasetParameters& dataset, const Config& config, PointCloud& ptCloud)
{
	std::cout << "Point Cloud Projection:\n\t";
	const size_t total = ptCloud.Z.size();

	if (ptCloud.Z.size() + ptCloud.px.size() + ptCloud.py.size() == 0) {
		setError("Point cloud was not initialized correctly.");
		return false;
	}

	// Camera constants
	const int spResFactor = config.superResolutionFactor;
	const float fxInv = 1.0f / (dataset.CAM_FX_px * spResFactor);
	const float ppx = dataset.CAM_PX_px * spResFactor;
	const float ppy = dataset.CAM_PY_px * spResFactor;

	// Find background invalid depth (Blender Scenes)
	const float bgVal = ptCloud.getMaxZ();

	// Prepare buffers with maximum possible size
	ptCloud.X.resize(total);
	ptCloud.Y.resize(total);

	// Get pointers for speed
	uint16_t* pxPtr = ptCloud.px.data();
	uint16_t* pyPtr = ptCloud.py.data();
	cv::Vec3b* cPtr = ptCloud.colors.data();

	float* Xptr = ptCloud.X.data();
	float* Yptr = ptCloud.Y.data();
	float* Zptr = ptCloud.Z.data(); // Already set

	// Single pass
	Eigen::Index validCount = 0;
	for (size_t i = 0; i < total; i++)
	{
		const float Z = Zptr[i];
		if (Z >= bgVal)  // skip invalid or zero depth
			continue;

		const float x = static_cast<float>(pxPtr[i]);
		const float y = static_cast<float>(pyPtr[i]);

		Zptr[validCount] = Z;
		Xptr[validCount] = (x - ppx) * Z * fxInv;
		Yptr[validCount] = (y - ppy) * Z * fxInv;

		// Keep only valid points
		pxPtr[validCount] = pxPtr[i];
		pyPtr[validCount] = pyPtr[i];
		cPtr[validCount] = cPtr[i];

		++validCount;
	}

	// Resize to valid size only once
	ptCloud.conservativeResize(validCount);

	std::cout << "* Point Cloud has " << validCount << " points.\n\t"
		<< "* Rejected Points: " << total - validCount << "\n\t" << std::endl;
}

bool PointCloudGeneration::adjustPointCloudToSystem(const SystemSpec& spec, const Config& config, PointCloud& ptCloud)
{
	std::cout << "Point Cloud Adjustments:\n\t";
	const size_t n = ptCloud.size();
	if (n == 0) return false;

	// ---------- 1. Compute stats ----------
	const PointCloudStats& stats = ptCloud.computeStats();

	const float cdp_mm = (spec.mla.displayDistance_mm * spec.mla.focalLength_mm) /
		(spec.mla.displayDistance_mm - spec.mla.focalLength_mm);

	float zOffset = 0.0f;
	switch (config.pointCloudMode) {
	case PointCloudMode::REAL:    if (stats.zMin <= 0.f) zOffset = stats.zMin; break;
	case PointCloudMode::VIRTUAL: if (stats.zMax >= 0.f) zOffset = -stats.zMax; break;
	case PointCloudMode::MLA:     zOffset = -stats.zCenter; break; // MLA at 0
	case PointCloudMode::CDP:     zOffset = cdp_mm; break;
	default: break;
	}

	// ---------- 2. Compute scale & offsets ----------
	if (stats.xRange == 0.f || stats.yRange == 0.f || stats.zRange == 0.f) {
		std::cerr << "Invalid X/Y range, skipping adjustments\n";
		return false;
	}

	const float scaleX = spec.display.width_mm / stats.xRange;
	const float scaleY = spec.display.height_mm / stats.yRange;
	const float xyScale = std::min(scaleX, scaleY);

	const float displayCenterX = spec.display.width_mm / 2.f;
	const float displayCenterY = spec.display.height_mm / 2.f;
	const float xOffset = displayCenterX - stats.xCenter;
	const float yOffset = displayCenterY - stats.yCenter;

	// Apply scale + offset
	float* X = ptCloud.X.data();
	float* Y = ptCloud.Y.data();
	float* Z = ptCloud.Z.data();

#pragma omp parallel for
	for (size_t i = 0; i < n; ++i) {
		X[i] = (X[i] - stats.xCenter) * xyScale + xOffset;
		Y[i] = (Y[i] - stats.yCenter) * xyScale + yOffset;
		Z[i] = (Z[i] - stats.zCenter) * xyScale + zOffset;
	}

	// ---------- 4. Print summary ----------
	std::cout << "* Scale factor: " << xyScale << "\n\t"
		<< "* XY scale : " << xyScale << "\n\t"
		<< "* Offset to display mode: " << PointCloudModeParser.toString(config.pointCloudMode) << "\n\t"
		<< "* CDP at: " << cdp_mm << "\n\t"
		<< "* X offset: " << xOffset << "\n\t"
		<< "* Y offset: " << yOffset << "\n\t"
		<< "* Z offset: " << zOffset << "\n\t"
		<< "* Display size: [" << spec.display.width_mm << " x " << spec.display.height_mm << "]\n";

	return true;
}

bool PointCloudGeneration::setupSteps()
{
	steps.clear();

	registerStep(
		"Initialize Point Cloud",
		[this](PipelineData& d) {
			return initPointCloud(d.rgbImage, d.depthMap, d.pointCloud);
		},
		true
	);

	registerStep(
		"Project 2D to 3D",
		[this](PipelineData& d) {
			return project2Dto3D(d.dataset, d.config, d.pointCloud);
		},
		true
	);

	registerStep(
		"Adjust Point Cloud To System",
		[this](PipelineData& d) {
			return adjustPointCloudToSystem(d.spec, d.config, d.pointCloud);
		},
		true
	);

	return true;
}