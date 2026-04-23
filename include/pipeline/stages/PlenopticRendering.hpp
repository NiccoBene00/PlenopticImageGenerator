#pragma once

#include "pipeline/PipelineStage.hpp"

class PlenopticRendering : public PipelineStage {
	PointCloud maskSelectZ0Range(const PointCloud& ptCloud, const float Z0Range_mm);
	std::unordered_map<int, std::vector<int>> groupPixelsPerML_linearIdx(const PointCloud& ptCloud, const MLASpec& mla);

	bool initMicroimageZ0(const PointCloud& ptCloud, const Config& config, const SystemSpec& spec, cv::Mat& plenopticImage);
	bool renderPlenopticImage(const SystemSpec& spec, PointCloud& ptCloud, cv::Mat& plenopticImage);

	bool setupSteps() override;
public:
	~PlenopticRendering() override = default;
	std::string getStageName() const override { return "Plenoptic Rendering"; }

};