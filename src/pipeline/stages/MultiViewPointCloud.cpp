

#include "pipeline/stages/MultiViewPointCloud.hpp"
#include "gpu/PointCloudGenerationGPU.cuh"
#include "pipeline/PipelineDataLoader.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>

// Pipeline of this stage: RGB + Depth (per camera) --> PointCloud (px, py, Z, color)
//                         --> GPU projection → (X, Y, Z) --> multiViewClouds[i]

/*
PIPELINE LOGIC OF THIS STAGE:

For each camera:
1. Load RGB image and depth map
2. Build a structure point cloud: (px, py, depth, color)
3. Send data to GPU
4. Backproject pixels into 3D coordinates (X, Y, Z)
5. Store resulting 3D point cloud in multiViewClouds[i]
*/


MultiViewPointCloud::MultiViewPointCloud() {
    
}

//  Helper: Convert RGB + Depth into image-space point cloud
void MultiViewPointCloud::fillPointCloudFromImages(PointCloud& ptCloud, const cv::Mat& rgb, const cv::Mat& depth)
{
    int width = depth.cols;
    int height = depth.rows;
    size_t total = width * height;

    // Allocate memory for all attributes
    ptCloud.Z.resize(total);
    ptCloud.px.resize(total);
    ptCloud.py.resize(total);
    ptCloud.colors.resize(total);

    /*
    Here we build a point cloud in IMAGE SPACE:
    - px, py = pixel coordinates
    - Z      = depth value
    - color  = RGB value

    No 3D projection yet --> this will happens later on GPU with the project2Dto3D()
    */
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

// Main GPU pipeline for multi-view point cloud generation
bool MultiViewPointCloud::generatePointCloudsGPU(
    PipelineData& data)
{
    if (data.calibration.empty() || data.multiViewClouds.empty()) {
        std::cerr << "[MultiViewPointCloud] Calibration or multiViewClouds not initialized\n";
        return false;
    }

    std::cout << "[MultiViewPointCloud] generatePointCloudsGPU called for "
              << data.calibration.size() << " cameras\n";

    //we process each camera independently
    for (size_t i = 0; i < data.calibration.size(); ++i) {
        std::string camId = "camera_" + std::to_string(i + 1);

        // Build file paths
        std::string rgbPath   = data.datasetPath + "/" + camId + ".png";
        std::string depthPath = data.datasetPath + "/" + camId + ".exr";

        std::cout << "[MultiViewPointCloud]\nCamera ID: " << camId 
                  << "\nRGB: " << rgbPath 
                  << "\nDepth: " << depthPath << "\n";

        
        //load rgb image
        cv::Mat rgb   = cv::imread(rgbPath, cv::IMREAD_COLOR);

        //load depth image (BUG WAS HERE!!!!!!!!!!)
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

        // Build image-space point cloud
        PointCloud& cloud = data.multiViewClouds[i];
        fillPointCloudFromImages(cloud, rgb, depth);

        //Prepare dataset parameters for this camera
        DatasetParameters camDataset = data.dataset;

        const CameraInfo& cam = data.calibration[i];

        // Focal length
        camDataset.CAM_FX_px = cam.focal_length_px;
        camDataset.CAM_FY_px = cam.focal_length_px; 

        // Principal point
        camDataset.CAM_PX_px = cam.principal_point_px[0];
        camDataset.CAM_PY_px = cam.principal_point_px[1];

        // Paths
        camDataset.rgbImagePath = rgbPath;
        camDataset.depthMapPath = depthPath;

        std::cout << "COORDINATES: " << camDataset.CAM_FX_px << "\n";

        // GPU projection: (px, py, Z) --> (X, Y, Z)
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

// Register pipeline step
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

