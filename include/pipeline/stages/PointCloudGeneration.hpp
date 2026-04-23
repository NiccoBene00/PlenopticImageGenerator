#pragma once

#include "pipeline/PipelineStage.hpp"

class PointCloudGeneration : public PipelineStage {
	//bool initPointCloud(const cv::Mat& rgbImage, const cv::Mat& depthMap, PointCloud& ptCloud);
	bool project2Dto3D(const DatasetParameters& dataset, const Config& config, PointCloud& ptCloud);
	bool adjustPointCloudToSystem(const SystemSpec& spec, const Config& config, PointCloud& ptCloud);

	bool setupSteps() override;
public:
	~PointCloudGeneration() override = default;
	bool initPointCloud(const cv::Mat& rgbImage, const cv::Mat& depthMap, PointCloud& ptCloud);
	std::string getStageName() const override { return "Point Cloud Generation"; }
};