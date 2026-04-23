#pragma once

#include <opencv2/opencv.hpp>

#include "DatasetParameters.hpp"
#include "SystemSpec.hpp"
#include "PointCloud.hpp"
#include "Config.hpp"

struct PipelineData {
	DatasetParameters dataset;
	SystemSpec spec;
	PointCloud pointCloud;
	Config config;

	std::string outputPath;

	cv::Mat inputRgbImage;
	cv::Mat inputDepthMap;
	cv::Mat rgbImage;
	cv::Mat depthMap;
	cv::Mat plenopticImage;
};
