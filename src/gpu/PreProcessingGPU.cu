#include "gpu/PreProcessingGPU.cuh"
#include "data/PipelineData.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace GPU {
namespace PreProcessingGPU {

// Helper CPU function for scaling
cv::Mat scaleMatCPU(const cv::Mat& input, int factor, cv::InterpolationFlags interp) {
    cv::Mat upscaled;
    cv::resize(input, upscaled, cv::Size(), factor, factor, interp);
    return upscaled;
}

bool superResolution(PipelineData& data) {
    const int factor = data.config.superResolutionFactor;

    if (data.inputRgbImage.empty()) {
        std::cerr << "[PreProcessingGPU] RGB image is empty before super-resolution.\n";
        return false;
    }

    if (data.inputDepthMap.empty()) {
        std::cerr << "[PreProcessingGPU] Depth map is empty before super-resolution.\n";
        return false;
    }

    // Scale RGB and depth images using CPU
    data.rgbImage   = scaleMatCPU(data.inputRgbImage, factor, cv::INTER_NEAREST);
    data.depthMap   = scaleMatCPU(data.inputDepthMap, factor, cv::INTER_NEAREST);

    return true;
}

} // namespace PreProcessingGPU
} // namespace GPU