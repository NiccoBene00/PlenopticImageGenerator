#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include "data/DatasetParameters.hpp"

class PipelineDataLoader {
	cv::Mat loadRGBImageFromFile(const std::string& rgbImagePath);
	cv::Mat loadDepthMapFromEXR(const std::string& exrDepthMapPath);
	cv::Mat loadDisparityMapFromYUV(const std::string& yuvDisparityMapPath, const int width, const int height, const int nBitsEncoded);

	cv::Mat convertMPEGDisparityToFloatMetric(const cv::Mat& depthMap, const int nBitEncoded, const float near_plane, const float far_plane);

public:
	PipelineDataLoader() = default;

	cv::Mat loadRGBImage(const std::string& rgbImagePath);
	cv::Mat loadDepthMap(const std::string& depthMapPath, const DatasetParameters& dataset);
};
