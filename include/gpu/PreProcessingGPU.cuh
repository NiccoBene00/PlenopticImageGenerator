#pragma once
#include <opencv2/core.hpp>

struct PipelineData;

namespace GPU {
namespace PreProcessing {

// CPU/GPU helper function declarations
cv::Mat superResolution(const cv::Mat& input, int scale);

} // namespace PreProcessing
} // namespace GPU