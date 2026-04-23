
#pragma once
#include "PointCloud.hpp"
#include <array>

//Helper function that applies Rotation Matrix 3x3 R and Translation Vector T 3x1 to all points in a PointCloud
//MEANING: R represents the orientation of the camera
//         T represents the position of the camera in the space
inline void applyRigidTransform(PointCloud& ptCloud,
                                const std::array<std::array<float,3>,3>& R,
                                const std::array<float,3>& T)
{
    
    for (size_t i = 0; i < ptCloud.size(); ++i) {
        float X = ptCloud.X[i];
        float Y = ptCloud.Y[i];
        float Z = ptCloud.Z[i];

        //P_global​ = R ⋅ P_local​ + T
        float Xnew = R[0][0]*X + R[0][1]*Y + R[0][2]*Z + T[0];
        float Ynew = R[1][0]*X + R[1][1]*Y + R[1][2]*Z + T[1];
        float Znew = R[2][0]*X + R[2][1]*Y + R[2][2]*Z + T[2];

        ptCloud.X[i] = Xnew;
        ptCloud.Y[i] = Ynew;
        ptCloud.Z[i] = Znew;
    }
}