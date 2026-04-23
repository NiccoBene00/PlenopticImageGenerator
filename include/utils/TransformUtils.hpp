#pragma once
#include <array>
#include <cmath>

//Converts Euler XYZ degrees --> rotation matrix

inline std::array<std::array<float,3>,3> eulerXYZtoMatrix(const std::array<float,3>& rotDeg) {
    float rx = rotDeg[0] * M_PI / 180.f;
    float ry = rotDeg[1] * M_PI / 180.f;
    float rz = rotDeg[2] * M_PI / 180.f;

    float cx = cos(rx), sx = sin(rx);
    float cy = cos(ry), sy = sin(ry);
    float cz = cos(rz), sz = sin(rz);

    std::array<std::array<float,3>,3> R;

    R[0][0] = cy*cz;
    R[0][1] = -cy*sz;
    R[0][2] = sy;

    R[1][0] = sx*sy*cz + cx*sz;
    R[1][1] = -sx*sy*sz + cx*cz;
    R[1][2] = -sx*cy;

    R[2][0] = -cx*sy*cz + sx*sz;
    R[2][1] = cx*sy*sz + sx*cz;
    R[2][2] = cx*cy;

    return R;
}