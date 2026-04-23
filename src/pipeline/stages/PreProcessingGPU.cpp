#include "pipeline/stages/PreProcessingGPU.hpp"
#include "data/PipelineData.hpp"
#include "gpu/PreProcessingGPU.cuh"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

// ---------------- GPU Helper Functions ----------------
namespace GPU {
namespace PreProcessing {

// CPU-only resize (GPU-ready interface)
cv::Mat superResolution(const cv::Mat& input, int scale) {
    if (input.empty()) return cv::Mat();

    cv::Mat output;
    cv::resize(input, output, cv::Size(), scale, scale, cv::INTER_NEAREST);
    return output;
}

} // namespace PreProcessing
} // namespace GPU

// ---------------- PreProcessingGPU Stage ----------------
bool PreProcessingGPU::superResolutionGPU(PipelineData& data) {
    const int factor = data.config.superResolutionFactor;

    if (data.inputRgbImage.empty()) {
        setError("RGB image is empty before super-resolution.");
        return false;
    }
    data.rgbImage = GPU::PreProcessing::superResolution(data.inputRgbImage, factor);

    if (data.inputDepthMap.empty()) {
        setError("Depth map is empty before super-resolution.");
        return false;
    }
    data.depthMap = GPU::PreProcessing::superResolution(data.inputDepthMap, factor);

    return true;
}

bool PreProcessingGPU::setupSteps() {
    steps.clear();

    registerStep(
        "Super Resolution GPU",
        [this](PipelineData& data) {
            return superResolutionGPU(data);
        }
    );

    return true;
}