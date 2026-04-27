#pragma once

#include <opencv2/opencv.hpp>

#include "DatasetParameters.hpp"
#include "SystemSpec.hpp"
#include "PointCloud.hpp"
#include "Config.hpp"
#include "CameraCalibration.hpp" 

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

	//// --- New fields for multi-view registration ---
	std::vector<PointCloud> multiViewClouds;       // One cloud per camera
    std::vector<CameraInfo> calibration;    // Calibration per camera
    PointCloud mergedCloud;           
	
	std::string datasetPath;             // Result of mergeAndDeduplicate

	bool isMultiView = false; // set automatically in main based on dataset path
};
