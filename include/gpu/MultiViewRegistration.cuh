#pragma once
#include "data/PipelineData.hpp"
#include "data/CameraCalibration.hpp"

namespace GPU {
namespace MultiViewRegistration {

    // load all point clouds from different cameras and apply rigid transforms
    bool loadAndTransformPointClouds(PipelineData& data);

    //merge all point clouds and remove duplicates with a certain tolerance
    bool mergeAndDeduplicate(PipelineData& data);

    

} // namespace MultiViewRegistration
} // namespace GPU