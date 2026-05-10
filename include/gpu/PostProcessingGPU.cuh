
#pragma once

#include <opencv2/opencv.hpp>
#include "data/SystemSpec.hpp"
#include "data/Config.hpp"
#include <vector>

namespace GPU {
namespace PostProcessing {

    struct MicroimageGPU {
        int x;
        int y;
        int width;
        int height;
    };


    // FUNCTIONS
    bool crackFiltering(
        cv::Mat& image,
        const std::vector<MicroimageGPU>& microimages,
        const Config& config
    );

    bool rotateMicroimages(cv::Mat& image, const SystemSpec& spec);

}
}