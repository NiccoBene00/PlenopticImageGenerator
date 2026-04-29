
#pragma once
#include "data/PipelineData.hpp"
#include "data/CameraCalibration.hpp"
#include <filesystem>
#include <iostream>

namespace DataLoader {

inline bool loadMultiViewDataset(const std::string& folder, PipelineData& data) {
    std::cout << "\n========================================================\n";
    std::filesystem::path datasetFolder(folder);
    if (!std::filesystem::exists(datasetFolder) || !std::filesystem::is_directory(datasetFolder)) {
        std::cerr << "[MultiViewDatasetLoader] Folder does not exist: " << folder << "\n";
        return false;
    }
    std::cout << "[MultiViewDatasetLoader] Found dataset folder: " << folder << "\n";

    // 1. Load camera calibration
    std::string calibFile = (datasetFolder / "camera_calibration.json").string();
    CameraCalibration calib;
    if (!calib.loadFromFile(calibFile)) {
        std::cerr << "[MultiViewDatasetLoader] Failed to load calibration: " << calibFile << "\n";
        return false;
    }
    data.calibration = calib.getAllCameras();
    std::cout << "[MultiViewDatasetLoader] Loaded camera calibration for " 
              << data.calibration.size() << " cameras\n";

    //print camera positions
    for (size_t i = 0; i < data.calibration.size(); ++i) {
        auto& cam = data.calibration[i];
        std::cout << "  Camera " << i+1 
                  << " position_mm: [" << cam.position_mm[0] << ", " 
                  << cam.position_mm[1] << ", " << cam.position_mm[2] << "]"
                  << " rotation_deg: [" << cam.rotation_xyz_deg[0] << ", "
                  << cam.rotation_xyz_deg[1] << ", " << cam.rotation_xyz_deg[2] << "]\n";
    }

    // Prepare multiViewClouds 
    data.multiViewClouds.resize(data.calibration.size());
    std::cout << "[MultiViewDatasetLoader] Initialized multiViewClouds vector with size: "
              << data.multiViewClouds.size() << "\n";

    //store paths in PipelineData
    data.datasetPath = folder;
    std::cout << "[MultiViewDatasetLoader] Dataset path set to: " << data.datasetPath << "\n";

    


    //------------------  DATASET PARAMETERS POPOLATION------------------
    if (!data.calibration.empty()) {
        const auto& cam0 = data.calibration[0];

        data.dataset.datasetName      = "multi_view_dataset";
        data.dataset.rgbImagePath     = ""; // multi-view uses single file
        data.dataset.depthMapPath     = ""; 
        data.dataset.depthEncoding    = DepthEncoding::FLOAT_METRIC; // default
        data.dataset.CAM_FX_px        = 2666.66;
        data.dataset.CAM_FY_px        = 1500; 
        data.dataset.CAM_PX_px        = 960;
        data.dataset.CAM_PY_px        = 540;
        data.dataset.nearPlane_m      = 0.1f; 
        data.dataset.farPlane_m       = 100.0f; 
        data.dataset.nBitsEncoded     = 0;
        
        cv::Mat depthImg = cv::imread((datasetFolder / "camera_1.exr").string(), cv::IMREAD_UNCHANGED);
        if (!depthImg.empty()) {
            data.dataset.depthWidth  = depthImg.cols;
            data.dataset.depthHeight = depthImg.rows;
            std::cout << "[MultiViewDatasetLoader] Depth image size set to: "
                    << data.dataset.depthWidth << "x" << data.dataset.depthHeight << "\n";
        }
    }
    std::cout << "========================================================\n";
    return true;
}

} // namespace DataLoader