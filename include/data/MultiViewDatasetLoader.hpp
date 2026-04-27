#pragma once
#include "data/PipelineData.hpp"
#include "data/CameraCalibration.hpp"
#include <filesystem>
#include <iostream>

namespace DataLoader {

inline bool loadMultiViewDataset(const std::string& folder, PipelineData& data) {
    std::filesystem::path datasetFolder(folder);
    if (!std::filesystem::exists(datasetFolder) || !std::filesystem::is_directory(datasetFolder)) {
        std::cerr << "[MultiViewDatasetLoader] Folder does not exist: " << folder << "\n";
        return false;
    }

    // 1. Load camera calibration
    std::string calibFile = (datasetFolder / "camera_calibration.json").string();
    CameraCalibration calib;
    if (!calib.loadFromFile(calibFile)) {
        std::cerr << "[MultiViewDatasetLoader] Failed to load calibration: " << calibFile << "\n";
        return false;
    }
    data.calibration = calib.getAllCameras();

    // 2. Prepare multiViewClouds (empty)
    data.multiViewClouds.resize(data.calibration.size());

    // 3. Store paths in PipelineData (optional, per stage to read later)
    data.datasetPath = folder;

    return true;
}

} // namespace DataLoader