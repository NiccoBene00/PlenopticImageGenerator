/*
#include "data/CameraCalibration.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>  



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
*/

#include "data/CameraCalibration.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
Logic of this file:
    - opens the camera_calibration.json file;
    - reads position_mm, rotation_xyz_deg, focal_length_px, and principal_point_px for each camera;
    - stores all camera info in a map cameras_ using camera name as key;
    - provides getCamera(name) to fetch parameters for any camera.
*/

bool CameraCalibration::loadFromFile(const std::string& filepath)
{
    std::ifstream file(filepath);
    if(!file.is_open()) {
        std::cerr << "Failed to open calibration file: " << filepath << std::endl;
        return false;
    }

    nlohmann::json j;
    file >> j;

    cameras_.clear();

    for(auto& [name, cam] : j.items()) {
        CameraInfo info;
        auto pos = cam["position_mm"];
        auto rot = cam["rotation_xyz_deg"];
        info.position_mm = { pos[0].get<float>(), pos[1].get<float>(), pos[2].get<float>() };
        info.rotation_xyz_deg = { rot[0].get<float>(), rot[1].get<float>(), rot[2].get<float>() };
        info.focal_length_px = cam["focal_length_px"].get<float>();
        auto pp = cam["principal_point_px"];
        info.principal_point_px = { pp[0].get<float>(), pp[1].get<float>() };

        // Convert Euler angles (XYZ, degrees) to rotation matrix
        float rx = info.rotation_xyz_deg[0] * M_PI / 180.0f;
        float ry = info.rotation_xyz_deg[1] * M_PI / 180.0f;
        float rz = info.rotation_xyz_deg[2] * M_PI / 180.0f;

        Eigen::Matrix3f Rx, Ry, Rz;
        Rx << 1,0,0, 0,cos(rx),-sin(rx), 0,sin(rx),cos(rx);
        Ry << cos(ry),0,sin(ry), 0,1,0, -sin(ry),0,cos(ry);
        Rz << cos(rz),-sin(rz),0, sin(rz),cos(rz),0, 0,0,1;

        info.rotationMatrix = Rz * Ry * Rx; // rotation order ZYX

        cameras_[name] = info;
    }

    return true;
}

const CameraInfo& CameraCalibration::getCamera(const std::string& name) const {
    return cameras_.at(name);
}

std::vector<CameraInfo> CameraCalibration::getAllCameras() const {
    std::vector<CameraInfo> result;
    for(auto& kv : cameras_)
        result.push_back(kv.second);
    return result;
}