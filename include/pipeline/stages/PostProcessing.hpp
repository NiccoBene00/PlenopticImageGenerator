#pragma once

#include "pipeline/PipelineStage.hpp"
#include <vector>

//-----new----
namespace GPU {
namespace PostProcessing {
    struct MicroimageGPU;
}
}
//-------------

class PostProcessing : public PipelineStage {
	struct MicroImageRegion { cv::Rect rect; int mi, mj; };
	std::vector<MicroImageRegion> microimages;

	bool filterCrackArtifacts(const SystemSpec& mla, const Config& config, cv::Mat& plenopticImage);
	bool rotateMicroimage180(cv::Mat& plenopticImage);
	bool pseudoscopicToOrthoscopic(cv::Mat& plenopticImage);
	bool smoothMicroimageEdges(cv::Mat& plenopticImage);

	bool setupSteps() override; 
public:
	~PostProcessing() override = default;
	std::string getStageName() const override { return "Post Processing"; }
	bool computeMicroimagesRegions(const SystemSpec& spec, const int imgWidth, const int imgHeight);

	//new
	std::vector<GPU::PostProcessing::MicroimageGPU> getMicroimagesGPU() const;

};