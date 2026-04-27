/*
#include "pipeline/stages/MultiViewPointCloud.hpp"
#include "data/CameraCalibration.hpp"
#include "gpu/PointCloudGenerationGPU.cuh"  
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <vector>
#include <string>

MultiViewPointCloud::MultiViewPointCloud(const CameraCalibration& calib)
    : calibration_(calib)
{}

// Helper: fill PointCloud from RGB and depth
void MultiViewPointCloud::fillPointCloudFromImages(PointCloud& ptCloud, const cv::Mat& rgb, const cv::Mat& depth)
{
    int width = depth.cols;
    int height = depth.rows;
    size_t total = width * height;

    ptCloud.Z.resize(total);
    ptCloud.px.resize(total);
    ptCloud.py.resize(total);
    ptCloud.colors.resize(total);

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

bool MultiViewPointCloud::generatePointCloudsGPU(
    const std::map<std::string, std::string>& rgbFiles,
    const std::map<std::string, std::string>& depthFiles,
    const DatasetParameters& dataset,
    const Config& config,
    std::vector<PointCloud>& cloudsOut)  
{
    cloudsOut.clear();

    for (const auto& [cameraId, rgbPath] : rgbFiles) {
        auto itDepth = depthFiles.find(cameraId);
        if (itDepth == depthFiles.end()) {
            std::cerr << "[MultiView] Missing depth for camera " << cameraId << "\n";
            continue;
        }

        cv::Mat rgb = cv::imread(rgbPath, cv::IMREAD_COLOR);
        if (rgb.empty()) {
            std::cerr << "[MultiView] Failed to read RGB: " << rgbPath << "\n";
            continue;
        }

        cv::Mat depth = cv::imread(itDepth->second, cv::IMREAD_UNCHANGED);
        if (depth.empty()) {
            std::cerr << "[MultiView] Failed to read depth: " << itDepth->second << "\n";
            continue;
        }

        if (rgb.size() != depth.size()) {
            std::cerr << "[MultiView] RGB and Depth size mismatch for camera " << cameraId << "\n";
            continue;
        }

        PointCloud ptCloud;
        fillPointCloudFromImages(ptCloud, rgb, depth);

        // GPU point cloud generation
        if (!GPU::PointCloudGPU::project2Dto3D(ptCloud, dataset, config)) {
            std::cerr << "[MultiView] Failed to generate point cloud for camera " << cameraId << "\n";
            continue;
        }

        cloudsOut.push_back(ptCloud);

        std::cout << "[MultiView] Point cloud generated for camera " << cameraId
                  << " --> points: " << ptCloud.size() << "\n";
    }

    return !cloudsOut.empty();
}

bool MultiViewPointCloud::setupSteps()
{
    steps.clear();

    registerStep(
        "Generate Multi-View Point Clouds GPU",
        [this](PipelineData& data) {
            // Use the new datasetPath member in PipelineData
            std::map<std::string, std::string> rgbFiles = {
                {"camera_1", data.datasetPath + "/camera_1.png"},
                {"camera_2", data.datasetPath + "/camera_2.png"},
                {"camera_3", data.datasetPath + "/camera_3.png"}
            };

            std::map<std::string, std::string> depthFiles = {
                {"camera_1", data.datasetPath + "/camera_1.exr"},
                {"camera_2", data.datasetPath + "/camera_2.exr"},
                {"camera_3", data.datasetPath + "/camera_3.exr"}
            };

            return generatePointCloudsGPU(rgbFiles, depthFiles, data.dataset, data.config, data.multiViewClouds);
        },
        true
    );

    return true;
}
*/

#include "pipeline/stages/MultiViewPointCloud.hpp"
#include "gpu/PointCloudGenerationGPU.cuh"
#include <opencv2/imgcodecs.hpp>
#include <iostream>

MultiViewPointCloud::MultiViewPointCloud(const std::vector<CameraInfo>& calib)
    : calibration_(calib)
{}

// Helper: fill PointCloud from RGB and depth
void MultiViewPointCloud::fillPointCloudFromImages(PointCloud& ptCloud, const cv::Mat& rgb, const cv::Mat& depth)
{
    int width = depth.cols;
    int height = depth.rows;
    size_t total = width * height;

    ptCloud.Z.resize(total);
    ptCloud.px.resize(total);
    ptCloud.py.resize(total);
    ptCloud.colors.resize(total);

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

bool MultiViewPointCloud::generatePointCloudsGPU(
    const std::map<std::string, std::string>& rgbFiles,
    const std::map<std::string, std::string>& depthFiles,
    const DatasetParameters& dataset,
    const Config& config,
    std::vector<PointCloud>& cloudsOut)
{
    cloudsOut.clear();

    for (const auto& [cameraId, rgbPath] : rgbFiles) {
        auto itDepth = depthFiles.find(cameraId);
        if (itDepth == depthFiles.end()) continue;

        cv::Mat rgb = cv::imread(rgbPath, cv::IMREAD_COLOR);
        cv::Mat depth = cv::imread(itDepth->second, cv::IMREAD_UNCHANGED);
        if (rgb.empty() || depth.empty() || rgb.size() != depth.size()) continue;

        PointCloud ptCloud;
        fillPointCloudFromImages(ptCloud, rgb, depth);

        if (!GPU::PointCloudGPU::project2Dto3D(ptCloud, dataset, config)) continue;

        cloudsOut.push_back(ptCloud);
    }

    return !cloudsOut.empty();
}

bool MultiViewPointCloud::setupSteps()
{
    steps.clear();

    registerStep(
        "Generate Multi-View Point Clouds GPU",
        [this](PipelineData& data) {
            std::map<std::string, std::string> rgbFiles, depthFiles;
            for (size_t i = 0; i < calibration_.size(); ++i) {
                std::string camId = "camera_" + std::to_string(i + 1);
                rgbFiles[camId]   = data.outputPath + "/" + camId + ".png";
                depthFiles[camId] = data.outputPath + "/" + camId + ".exr";
            }
            return generatePointCloudsGPU(rgbFiles, depthFiles, data.dataset, data.config, data.multiViewClouds);
        },
        true
    );

    return true;
}