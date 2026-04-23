#pragma once

#include <opencv2/opencv.hpp>

#include "data/PointCloud.hpp"
#include "data/SystemSpec.hpp"

namespace GPU {
	namespace Render {
		bool renderPlenopticImage(cv::Mat& plenopticImage, PointCloud& ptCloud, const SystemSpec& spec, std::string& errorMsg);
	}
}