#include "data/CameraCalibration.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>  

/*
Logic of this file:
    - opens the camera_calibration.json file;
    - reads position_mm, rotation_xyz_deg, focal_length_px, and principal_point_px for each camera;
    - stores all camera info in a map cameras_ using camera name as key;
    - provides getCamera(name) to fetch parameters for any camera.
*/

bool CameraCalibration::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[CameraCalibration] Failed to open file: " << filepath << std::endl;
        return false;
    }

    nlohmann::json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        std::cerr << "[CameraCalibration] JSON parse error: " << e.what() << std::endl;
        return false;
    }

    for (auto& [name, cam] : j.items()) {
        CameraInfo info;

        if (cam.contains("position_mm") && cam["position_mm"].is_array()) {
            for (size_t i = 0; i < 3; ++i)
                info.position_mm[i] = cam["position_mm"][i].get<float>();
        }

        if (cam.contains("rotation_xyz_deg") && cam["rotation_xyz_deg"].is_array()) {
            for (size_t i = 0; i < 3; ++i)
                info.rotation_xyz_deg[i] = cam["rotation_xyz_deg"][i].get<float>();
        }

        if (cam.contains("focal_length_px"))
            info.focal_length_px = cam["focal_length_px"].get<float>();

        if (cam.contains("principal_point_px") && cam["principal_point_px"].is_array()) {
            info.principal_point_px[0] = cam["principal_point_px"][0].get<float>();
            info.principal_point_px[1] = cam["principal_point_px"][1].get<float>();
        }

        cameras_[name] = info;
    }

    return true;
}

const CameraInfo& CameraCalibration::getCamera(const std::string& name) const {
    auto it = cameras_.find(name);
    if (it == cameras_.end()) {
        throw std::runtime_error("[CameraCalibration] Camera not found: " + name);
    }
    return it->second;
}