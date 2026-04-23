#include "pipeline/stages/PreProcessing.hpp"

#include <iostream>

cv::Mat PreProcessing::scaleMat(const int scale, const cv::InterpolationFlags interpolation, cv::Mat& data) {
	cv::Mat upscaled;
	cv::resize(data, upscaled, cv::Size(), scale, scale, interpolation);
	return upscaled;
}

bool PreProcessing::superResolution(PipelineData& data) {
	const int factor = data.config.superResolutionFactor;

	// Scaling RGB image
	if (data.inputRgbImage.empty()) {
		setError("RGB image is empty before super - resolution.");
		return false;
	}
	cv::Mat upscaledRGB = scaleMat(factor, cv::INTER_NEAREST, data.inputRgbImage);
	if (upscaledRGB.empty()) {
		setError("Failed to upscale RGB image.");
		return false;
	}
	data.rgbImage = upscaledRGB;

	// Scaling Depth Map
	if (data.inputDepthMap.empty()) {
		setError("Depth map is empty before super-resolution.");
		return false;
	}
	cv::Mat upscaledDepth = scaleMat(factor, cv::INTER_NEAREST, data.inputDepthMap);
	if (upscaledDepth.empty()) {
		setError("Failed to upscale depth map.");
		return false;
	}
	data.depthMap = upscaledDepth;

	return true;
}

bool PreProcessing::setupSteps()
{
	steps.clear();

	registerStep(
		"Super Resolution",
		[this](PipelineData& data) {
			return superResolution(data);
		}
	);

	return true;
}


