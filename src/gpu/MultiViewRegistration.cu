
#include "gpu/MultiViewRegistration.cuh"
#include "gpu/PointCloudGenerationGPU.cuh"
#include "data/CameraCalibration.hpp"
#include "data/PipelineData.hpp"
#include "gpu/GPUTypes.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

//PIPELINE of this stage: (Point cloud per camera, coordinate camera) --> [transformKernel] := world coordinates
//                          --> [merge] := unique cloud --> deduplicate := overlap removal



#define CUDA_CHECK(call)                                           \
do {                                                               \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(err)      \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";\
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)

namespace GPU {
namespace MultiViewRegistration {

// -------------------- CUDA KERNELS --------------------

/*
This CUDA kernels applies the following logic:

Input:
- (X, Y, Z): point in camera coordinates
- R: 3x3 rotation matrix
- t: translation vector

Output:
- transformed point in world coordinates

Formula:
p_world = R^(-1) * p_cam + C
*/
__global__ void transformKernel(
    float* X, float* Y, float* Z,
    const float* R, const float* t,
    size_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float xi = X[i];
    float yi = Y[i];
    float zi = Z[i];

    //Rigid Body Transformation
    float Xnew = R[0]*xi + R[1]*yi + R[2]*zi + t[0];
    float Ynew = R[3]*xi + R[4]*yi + R[5]*zi + t[1];
    float Znew = R[6]*xi + R[7]*yi + R[8]*zi + t[2];

    X[i] = Xnew;
    Y[i] = Ynew;
    Z[i] = Znew;
}

// STILL TO IMPLEMENT: Mark duplicates within tolerance 
__global__ void markDuplicateKernel(
    float* X, float* Y, float* Z,
    unsigned char* mask,
    size_t N,
    float tolerance)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !mask[i]) return;

    float xi = X[i];
    float yi = Y[i];
    float zi = Z[i];

    //here i check for each point i if the comparison with the point j and if the 
    //AABB distance (cube) between them is smaller than a tolerance
    /*
    for (size_t j = i + 1; j < N; ++j) {
        if (!mask[j]) continue; //more threads can overwirte here????
        float dx = fabs(xi - X[j]);
        float dy = fabs(yi - Y[j]);
        float dz = fabs(zi - Z[j]);
        if (dx <= tolerance && dy <= tolerance && dz <= tolerance)
            mask[j] = 0;
    }
    */
    //MAYBE should be better taking one ref point and make all other threads analyze this
}

// -------------------- HOST FUNCTIONS --------------------

/*
For each camera:
1. Take point cloud in camera coordinates
2. Upload to GPU
3. Apply extrinsic transformation (R, t)
4. Convert all points into world coordinates
*/

bool loadAndTransformPointClouds(PipelineData& data)
{
    const int blockSize = 256;

    for(int camIdx = 0; camIdx < 3; ++camIdx) {
        auto& cloud = data.multiViewClouds[camIdx];

        //get CameraInfo from CameraCalibration
        CameraInfo& calib = data.calibration[camIdx];

        std::cout << "CAM POS: " << calib.position_mm[0] << ", " << calib.position_mm[1] << ", " <<
                calib.position_mm[2] << "\n";

        size_t N = cloud.size();
        if(N == 0) continue; //skip empty point clouds

        // Allocate GPU buffers for 3D coordinates
        float *d_X, *d_Y, *d_Z;
        CUDA_CHECK(cudaMalloc(&d_X, N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Y, N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Z, N*sizeof(float)));

        // Transfer point cloud coordinates from CPU to GPU
        CUDA_CHECK(cudaMemcpy(d_X, cloud.X.data(), N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Y, cloud.Y.data(), N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Z, cloud.Z.data(), N*sizeof(float), cudaMemcpyHostToDevice));

        // Convert Eigen rotation matrix into a flat GPU-compatible array
        // Flatten rotation matrix and translation vector
        float R[9];
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                R[3*r + c] = calib.rotationMatrix(r,c); //conversion from Eigen to array flat structure GPU-ready
        
            
        // Compute inverse rotation matrix
        float Rinv[9];
        for(int r=0;r<3;r++)
            for(int c=0;c<3;c++)
                Rinv[3*r + c] = R[3*c + r];

        
        // Camera center position in world coordinates
        float t[3] = { calib.position_mm[0], calib.position_mm[1], calib.position_mm[2] };

        // Upload transformation parameters to GPU
        float *d_R, *d_t;
        CUDA_CHECK(cudaMalloc(&d_R, 9*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_t, 3*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_R, Rinv, 9*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_t, t, 3*sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch CUDA transformation kernel
        //
        // Each thread transforms one 3D point independently:
        //
        //      p_world = R^(-1) * p_camera + C
        int gridSize = (N + blockSize - 1)/blockSize;
        transformKernel<<<gridSize, blockSize>>>(d_X, d_Y, d_Z, d_R, d_t, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(cloud.X.data(), d_X, N*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cloud.Y.data(), d_Y, N*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cloud.Z.data(), d_Z, N*sizeof(float), cudaMemcpyDeviceToHost));

        cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Z);
        cudaFree(d_R); cudaFree(d_t);
    }

    return true;
}

/*
Merge step:
1. Concatenate all transformed point clouds
2. Optionally remove duplicates (currently to finish)
3. Build final unified point cloud
*/

bool mergeAndDeduplicate(PipelineData& data)
{
    if (data.multiViewClouds.size() != 3) {
        std::cerr << "[Error] Expecting three clouds.\n";
        return false;
    }

    // Compute total number of points after concatenation
    size_t totalPoints = 0;
    for (auto& c : data.multiViewClouds)
        totalPoints += c.size();


    // Allocate concatenated host arrays
    //
    // The three point clouds are merged into a single linear structure:
    //
    //      cloud1 + cloud2 + cloud3
    std::vector<float> X(totalPoints), Y(totalPoints), Z(totalPoints);
    std::vector<cv::Vec3b> colors(totalPoints);
    
    //concatenate all clouds into unified arrays
    size_t offset = 0;
    for (auto& c : data.multiViewClouds) {
        std::copy(c.X.begin(), c.X.end(), X.begin() + offset);
        std::copy(c.Y.begin(), c.Y.end(), Y.begin() + offset);
        std::copy(c.Z.begin(), c.Z.end(), Z.begin() + offset);
        std::copy(c.colors.begin(), c.colors.end(), colors.begin() + offset);
        offset += c.size();
    }

    //GPU buffers
    float *d_X, *d_Y, *d_Z;
    unsigned char *d_mask;
    CUDA_CHECK(cudaMalloc(&d_X, totalPoints * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, totalPoints * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z, totalPoints * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, totalPoints * sizeof(unsigned char)));

    CUDA_CHECK(cudaMemcpy(d_X, X.data(), totalPoints * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, Y.data(), totalPoints * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Z, Z.data(), totalPoints * sizeof(float), cudaMemcpyHostToDevice));
    //initialize validity mask
    //
    // Default:
    //      1 -> valid point
    CUDA_CHECK(cudaMemset(d_mask, 1, totalPoints * sizeof(unsigned char)));

    // Launch duplicate detection kernel
    //
    // Each thread compares one point against neighboring points and
    // invalidates duplicates using a spatial tolerance threshold
    int blockSize = 256;
    int gridSize = (totalPoints + blockSize - 1) / blockSize;
    markDuplicateKernel<<<gridSize, blockSize>>>(d_X, d_Y, d_Z, d_mask, totalPoints, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy validity mask back to CPU
    std::vector<unsigned char> mask(totalPoints);
    CUDA_CHECK(cudaMemcpy(mask.data(), d_mask, totalPoints * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    //count remaining valid points
    size_t validCount = std::count(mask.begin(), mask.end(), 1);
    
    // Allocate final merged point cloud
    data.mergedCloud.X.resize(validCount);
    data.mergedCloud.Y.resize(validCount);
    data.mergedCloud.Z.resize(validCount);
    data.mergedCloud.colors.resize(validCount);

    // Build compact merged cloud
    // Only points marked as valid are preserved
    for (size_t idx = 0, out = 0; idx < totalPoints; ++idx) {
        if (mask[idx]) {
            data.mergedCloud.X[out] = X[idx];
            data.mergedCloud.Y[out] = Y[idx];
            data.mergedCloud.Z[out] = Z[idx];
            data.mergedCloud.colors[out] = colors[idx];
            ++out;
        }
    }

    
    GPU::PointCloudGPU::adjustToSystem(
        data.mergedCloud,
        data.spec,
        data.config
    );

    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Z);
    cudaFree(d_mask);

    //store point cloud for rendering stage
    data.pointCloud = data.mergedCloud;
    std::cout << "[GPU] Merged points count: " << data.pointCloud.size() << "\n";
    std::cout << "[GPU] Merged points color count: " << data.pointCloud.colors.size() << "\n";

    return true;
}

} // namespace MultiViewRegistration
} // namespace GPU