
#pragma once

#include "data/PipelineData.hpp"
#include "pipeline/PipelineStage.hpp"
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

// -------------------- Multi-View Point Cloud Stage --------------------
// Gestisce la generazione di point cloud per dataset multi-view
class MultiViewPointCloud : public PipelineStage {
public:

    MultiViewPointCloud();

    // Genrate GPU point clouds for each camera in PipelineData
    bool generatePointCloudsGPU(PipelineData& data);

    // To give access to each point cloud
    const std::map<std::string, PointCloud>& getPointClouds() const { return clouds_; }

    std::string getStageName() const override { return "MultiView Point Cloud GPU"; }

protected:
    bool setupSteps() override;

private:
    std::map<std::string, PointCloud> clouds_; // map: camera_id -> PointCloud

    //Helper to populate a Point Cloud from img rgb+depth
    static void fillPointCloudFromImages(PointCloud& ptCloud, const cv::Mat& rgb, const cv::Mat& depth);
};