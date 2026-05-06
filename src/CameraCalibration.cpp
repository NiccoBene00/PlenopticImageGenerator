
#include "data/CameraCalibration.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
GENERAL LOGIC:

1. Open the calibration JSON file.
2. Parse all camera entries.
3. For each camera:
   - Read position, rotation, focal length, and principal point.
   - Convert Euler angles (degrees) to radians.
   - Build rotation matrices (Rx, Ry, Rz).
   - Combine them into a single rotation matrix.
4. Store everything in a map for fast access.
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

        //load position
        auto pos = cam["position_mm"];
        info.position_mm = { pos[0].get<float>(), pos[1].get<float>(), pos[2].get<float>() };

        //load Euler rotations (degrees)
        auto rot = cam["rotation_xyz_deg"];
        info.rotation_xyz_deg = { rot[0].get<float>(), rot[1].get<float>(), rot[2].get<float>() };

        //load intrinsic parameters
        info.focal_length_px = cam["focal_length_px"].get<float>();
        info.focal_length_py = cam["focal_length_py"].get<float>();
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

        info.rotationMatrix = Rz * Ry * Rx; // convention: rotation order ZYX (from right to left)


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