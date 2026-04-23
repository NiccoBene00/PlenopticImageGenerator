#pragma once

#include "pipeline/PipelineStage.hpp"

class PreProcessing : public PipelineStage {
	// TODO: test other interpolation methods: depthmap must be discontinuous
	cv::Mat scaleMat(const int scale, const cv::InterpolationFlags interpolationMethod, cv::Mat& data);
	bool superResolution(PipelineData& data);

	bool setupSteps() override;
public:
	~PreProcessing() override = default;
	std::string getStageName() const override { return "Pre Processing"; }
};