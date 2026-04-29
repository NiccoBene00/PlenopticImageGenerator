/*
#pragma once

#include "data/CameraCalibration.hpp"
#include "data/DatasetParameters.hpp"
#include "data/Config.hpp"
#include "data/PointCloud.hpp"
#include "pipeline/PipelineStage.hpp"
#include <string>
#include <vector>
#include <map>

// Manages multiple point clouds from a multi-view dataset
class MultiViewPointCloud : public PipelineStage {
public:
    MultiViewPointCloud(const CameraCalibration& calib);

    // Generate GPU point clouds for all cameras and store in the provided vector
    bool generatePointCloudsGPU(const std::map<std::string, std::string>& rgbFiles,
                                const std::map<std::string, std::string>& depthFiles,
                                const DatasetParameters& dataset,
                                const Config& config,
                                std::vector<PointCloud>& cloudsOut);

    // Access generated clouds
    const std::map<std::string, PointCloud>& getPointClouds() const { return clouds_; }

protected:
    bool setupSteps() override;

private:
    const CameraCalibration& calibration_;
    std::map<std::string, PointCloud> clouds_; // map camera_id --> point cloud

    // helper to fill PointCloud from RGB+Depth images
    static void fillPointCloudFromImages(PointCloud& ptCloud, const cv::Mat& rgb, const cv::Mat& depth);
};
*/

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
    // Costruttore vuoto, non serve passare calibration esterna
    MultiViewPointCloud();

    // Genera GPU point clouds per tutte le camere presenti in PipelineData
    bool generatePointCloudsGPU(PipelineData& data);

    // Accesso alle point cloud generate (opzionale)
    const std::map<std::string, PointCloud>& getPointClouds() const { return clouds_; }

    std::string getStageName() const override { return "MultiView Point Cloud GPU"; }

protected:
    bool setupSteps() override;

private:
    std::map<std::string, PointCloud> clouds_; // map: camera_id -> PointCloud

    // Helper per popolare un PointCloud da immagini RGB+Depth
    static void fillPointCloudFromImages(PointCloud& ptCloud, const cv::Mat& rgb, const cv::Mat& depth);
};