#pragma once

#include <string>
#include <vector>
#include <array>
#include <unordered_map>

struct CameraInfo {
    std::array<float, 3> position_mm;
    std::array<float, 3> rotation_xyz_deg;
    float focal_length_px;
    std::array<float, 2> principal_point_px;
};

class CameraCalibration {
public:
    bool loadFromFile(const std::string& filepath);

    const CameraInfo& getCamera(const std::string& name) const;

private:
    std::unordered_map<std::string, CameraInfo> cameras_;
};