#pragma once

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <Eigen/Dense>

/*
This class manages a set of calibrated cameras

Functionalities:
- Load camera parameters from a JSON file
- Store them in a map (camera name -> CameraInfo)
- Provide access to individual cameras or all cameras
*/

struct CameraInfo {
    std::array<float, 3> position_mm;
    std::array<float, 3> rotation_xyz_deg;
    float focal_length_px;
    float focal_length_py;
    std::array<float, 2> principal_point_px;

    Eigen::Matrix3f rotationMatrix;
};

class CameraCalibration {
public:
    bool loadFromFile(const std::string& filepath);

    const CameraInfo& getCamera(const std::string& name) const;

    std::vector<CameraInfo> getAllCameras() const;

private:
    std::unordered_map<std::string, CameraInfo> cameras_;
};


