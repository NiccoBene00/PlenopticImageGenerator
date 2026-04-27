#pragma once

#include "data/CameraCalibration.hpp"
#include "data/DatasetParameters.hpp"
#include "data/Config.hpp"
#include "data/PointCloud.hpp"
#include "pipeline/PipelineStage.hpp"
#include <string>
#include <vector>
#include <map>

// Manages multiple point clouds from multi-view dataset
class MultiViewPointCloud : public PipelineStage {
public:
    MultiViewPointCloud(const CameraCalibration& calib);

    // Generate GPU point clouds for all cameras
    bool generatePointCloudsGPU(const std::map<std::string, std::string>& rgbFiles,
                                const std::map<std::string, std::string>& depthFiles,
                                const DatasetParameters& dataset,
                                const Config& config);

    // Access generated clouds
    const std::map<std::string, PointCloud>& getPointClouds() const { return clouds_; }

protected:
    bool setupSteps() override;

private:
    const CameraCalibration& calibration_;
    std::map<std::string, PointCloud> clouds_; // map camera_id --> point cloud
};