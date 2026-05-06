
#pragma once
#include "data/PipelineData.hpp"
#include "data/CameraCalibration.hpp"
#include <filesystem>
#include <iostream>

namespace DataLoader {

/*
This function loads a multi-view dataset from a folder and prepares all the data
needed for the reconstruction pipeline.

Main steps:
1. Validate dataset folder
2. Load camera calibration (extrinsics + intrinsics)
3. Initialize per-view point cloud containers
4. Store dataset  parameters (camera intrinsics, depth format, etc.)
*/

inline bool loadMultiViewDataset(const std::string& folder, PipelineData& data) {
    std::cout << "\n========================================================\n";

    //1. Validate dataset folde
    std::filesystem::path datasetFolder(folder);
    if (!std::filesystem::exists(datasetFolder) || !std::filesystem::is_directory(datasetFolder)) {
        std::cerr << "[MultiViewDatasetLoader] Folder does not exist: " << folder << "\n";
        return false;
    }
    std::cout << "[MultiViewDatasetLoader] Found dataset folder: " << folder << "\n";

    // 2. Load camera calibration
    std::string calibFile = (datasetFolder / "camera_calibration.json").string();

    CameraCalibration calib;
    if (!calib.loadFromFile(calibFile)) {
        std::cerr << "[MultiViewDatasetLoader] Failed to load calibration: " << calibFile << "\n";
        return false;
    }

    ///Store all cameras inside pipeline data
    data.calibration = calib.getAllCameras();
    std::cout << "[MultiViewDatasetLoader] Loaded camera calibration for " 
              << data.calibration.size() << " cameras\n";


    //DEBUG: print camera positions
    for (size_t i = 0; i < data.calibration.size(); ++i) {
        auto& cam = data.calibration[i];
        std::cout << "  Camera " << i+1 
                  << " position_mm: [" << cam.position_mm[0] << ", " 
                  << cam.position_mm[1] << ", " << cam.position_mm[2] << "]"
                  << " rotation_deg: [" << cam.rotation_xyz_deg[0] << ", "
                  << cam.rotation_xyz_deg[1] << ", " << cam.rotation_xyz_deg[2] << "]\n";
    }

    // 3. Initialize multi-view point cloud containers
    // Each camera will generate its own point cloud
    data.multiViewClouds.resize(data.calibration.size());
    std::cout << "[MultiViewDatasetLoader] Initialized multiViewClouds vector with size: "
              << data.multiViewClouds.size() << "\n";

    
    //4. store paths in PipelineData
    data.datasetPath = folder;
    std::cout << "[MultiViewDatasetLoader] Dataset path set to: " << data.datasetPath << "\n";

    


    //------------------  DATASET PARAMETERS POPOLATION------------------
    if (!data.calibration.empty()) {
        const auto& cam0 = data.calibration[0];

        data.dataset.datasetName      = "multi_view_dataset";
        data.dataset.rgbImagePath     = ""; 
        data.dataset.depthMapPath     = ""; 
        data.dataset.depthEncoding    = DepthEncoding::FLOAT_METRIC;
        data.dataset.CAM_FX_px        = cam0.focal_length_px;
        data.dataset.CAM_FY_px        = cam0.focal_length_py;
        data.dataset.CAM_PX_px        = cam0.principal_point_px[0];
        data.dataset.CAM_PY_px        = cam0.principal_point_px[1];
        data.dataset.nearPlane_m      = 0.1f; 
        data.dataset.farPlane_m       = 100.0f; 
        data.dataset.nBitsEncoded     = 0;
        
        // Load one depth image to infer resolution
        cv::Mat depthImg = cv::imread((datasetFolder / "camera_1.exr").string(), cv::IMREAD_UNCHANGED);
        if (!depthImg.empty()) {
            data.dataset.depthWidth  = depthImg.cols;
            data.dataset.depthHeight = depthImg.rows;
            std::cout << "[MultiViewDatasetLoader] Depth image size set to: "
                    << data.dataset.depthWidth << "x" << data.dataset.depthHeight << "\n";
        }

        // Debug info about depth format
        std::cout << "Depth Type: " << depthImg.type() << "\n";
        std::cout << "Depth Channels: " << depthImg.channels() << std::endl;
    }
    std::cout << "========================================================\n";
    return true;
}

} // namespace DataLoader