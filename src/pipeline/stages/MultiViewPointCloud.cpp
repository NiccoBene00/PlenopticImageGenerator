

#include "pipeline/stages/MultiViewPointCloud.hpp"
#include "gpu/PointCloudGenerationGPU.cuh"
#include "pipeline/PipelineDataLoader.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>

// Pipeline of this stage: RGB + Depth (per camera) --> PointCloud (px, py, Z, color)
//                         --> GPU projection → (X, Y, Z reali) --> multiViewClouds[i]

MultiViewPointCloud::MultiViewPointCloud() {
    
}

// -------------------- Helper: Fill PointCloud from RGB+Depth --------------------
void MultiViewPointCloud::fillPointCloudFromImages(PointCloud& ptCloud, const cv::Mat& rgb, const cv::Mat& depth)
{
    int width = depth.cols;
    int height = depth.rows;
    size_t total = width * height;

    ptCloud.Z.resize(total);
    ptCloud.px.resize(total);
    ptCloud.py.resize(total);
    ptCloud.colors.resize(total);

    //here basically I built a point cloud in image coordinates: (px,py), Z, color
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = y * width + x;
            ptCloud.px[idx] = x;
            ptCloud.py[idx] = y;
            ptCloud.Z[idx] = depth.at<float>(y, x);
            cv::Vec3b c = rgb.at<cv::Vec3b>(y, x);
            ptCloud.colors[idx] = { c[0], c[1], c[2] };
        }
    }
}

// -------------------- GPU PointCloud Generation --------------------
bool MultiViewPointCloud::generatePointCloudsGPU(
    PipelineData& data)
{
    if (data.calibration.empty() || data.multiViewClouds.empty()) {
        std::cerr << "[MultiViewPointCloud] Calibration or multiViewClouds not initialized\n";
        return false;
    }

    std::cout << "[MultiViewPointCloud] generatePointCloudsGPU called for "
              << data.calibration.size() << " cameras\n";

    for (size_t i = 0; i < data.calibration.size(); ++i) {
        std::string camId = "camera_" + std::to_string(i + 1);

        std::string rgbPath   = data.datasetPath + "/" + camId + ".png";
        std::string depthPath = data.datasetPath + "/" + camId + ".exr";

        std::cout << "[MultiViewPointCloud]\nCamera ID: " << camId 
                  << "\nRGB: " << rgbPath 
                  << "\nDepth: " << depthPath << "\n";

        cv::Mat rgb   = cv::imread(rgbPath, cv::IMREAD_COLOR);
        //cv::Mat depth = cv::imread(depthPath, cv::IMREAD_UNCHANGED);
        PipelineDataLoader loader;
        cv::Mat depth = loader.loadDepthMap(depthPath, data.dataset);
        std::cout << "DEPTH CHANNELS: " << depth.channels() << std::endl;
        
        if (rgb.empty()) {
            std::cerr << "[MultiViewPointCloud] Failed to load RGB: " << rgbPath << "\n";
            continue;
        }
        if (depth.empty()) {
            std::cerr << "[MultiViewPointCloud] Failed to load Depth: " << depthPath << "\n";
            continue;
        }
        if (rgb.size() != depth.size()) {
            std::cerr << "[MultiViewPointCloud] RGB/Depth size mismatch for " << camId
                      << " RGB: " << rgb.cols << "x" << rgb.rows
                      << " Depth: " << depth.cols << "x" << depth.rows << "\n";
            continue;
        }

        PointCloud& cloud = data.multiViewClouds[i];
        fillPointCloudFromImages(cloud, rgb, depth);

        // -------------------- Dataset building for the camera --------------------
        DatasetParameters camDataset = data.dataset;
        camDataset.rgbImagePath = rgbPath;
        camDataset.depthMapPath = depthPath;

        std::cout << "COORDINATES: " << camDataset.CAM_FX_px << "\n";
        // -------------------- GPU projection --------------------
        if (!GPU::PointCloudGPU::project2Dto3D(cloud, camDataset, data.config)) {
            std::cerr << "[MultiViewPointCloud] GPU project2Dto3D failed for " << camId << "\n";
            continue;
        }

        std::cout << "[MultiViewPointCloud] Camera " << camId 
                  << " point cloud size: " << cloud.size() << "\n";
        std::cout << "\n";
    }

    return true;
}

// -------------------- Setup Steps --------------------
bool MultiViewPointCloud::setupSteps()
{
    steps.clear();

    registerStep(
        "Generate Multi-View Point Clouds GPU",
        [this](PipelineData& data) {
            return generatePointCloudsGPU(data);
        },
        true
    );

    return true;
}

