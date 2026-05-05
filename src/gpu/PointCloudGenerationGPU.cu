

//PIPELINE POINT CLOUD GENERATION
// RGB + DEPTH -> initPointCloud -> (px, py, Z, color) -> project2Dto3D -> (X, Y, Z, filtered) -> adjustPointCloud 
// -> coordinates in display system

//Firs initiPointCloud allows to go from an image 2D to a 1D array of point like (x_img, y_img, depth, RGB color)
//Here the point are still independent, we dont have any kind of coordinates system

//project2Dto3D aims to convert from (pixel + depth) to real 3D coordinates
//It uses the pinhole camera model like follow
//  X = (x - cx) * Z / fx      where (x - cx) and (y - cy) means how far the pixel is from the optical center
//  Y = (y - cy) * Z / fy
//  Z = Z
//During this stage we need to deal with invalid point (like infinite depth or zero depth or background point); How?
//We remove them.
//At the end of this stage we have a 3D point in the camera space where 
//  origin -> camera
//  Z axis -> depth
//  X, Y   -> image plane projected

//adjustPointCloudToSystem now aims to go from camera system to display phsical system
//This stage performs operations like
//  centering (point cloud center -> display center)
//  scaling (keep the proportion to fit in the display)
//  Z offset (to set the scene compared to the display and the microlens, here we have 4 
//            modalities like REAL, VIRTUAL, MLA, CDP)
// Final output -> (X, Y, Z) -> coordinates in mm



#include "gpu/PointCloudGenerationGPU.cuh"
#include "gpu/GPUTypes.cuh"
#include "data/PointCloud.hpp"
#include "data/SystemSpec.hpp"
#include "data/Config.hpp"
#include "data/DatasetParameters.hpp"
#include <opencv2/core.hpp>

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

struct PointPacked
{
    float X, Y, Z;
    uint16_t px, py;
    RGB8 color;
};

#define CUDA_CHECK(call)                                           \
do {                                                               \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(err)      \
                  << " at " << __FILE__ << ":" << __LINE__          \
                  << std::endl;                                    \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)


namespace GPU {
namespace PointCloudGPU {

//=================================================
__global__ void computeMaskKernel(
    const float* __restrict__ Z,
    int* __restrict__ mask,
    float bgVal,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float z = __ldg(&Z[i]);   // read-only cache
    mask[i] = (z < bgVal) ? 1 : 0;
}


__global__ void projectScatterKernel(
    const float* __restrict__ Z,
    const uint16_t* __restrict__ px,
    const uint16_t* __restrict__ py,
    const RGB8* __restrict__ colors,

    float* __restrict__ Xout,
    float* __restrict__ Yout,
    float* __restrict__ Zout,
    uint16_t* __restrict__ pxOut,
    uint16_t* __restrict__ pyOut,
    RGB8* __restrict__ colOut,

    const int* __restrict__ mask,
    const int* __restrict__ scan,

    float fxInv,
    float ppx,
    float ppy,
    int spResFactor,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // load mask first to avoid useless load
    int m = __ldg(&mask[i]);
    if (m == 0) return;

    int outIdx = __ldg(&scan[i]) - 1;

    // load once (register caching)
    float Zval = __ldg(&Z[i]);
    uint16_t pxv = __ldg(&px[i]);
    uint16_t pyv = __ldg(&py[i]);
    RGB8 col = colors[i];

    float x = (float)pxv * spResFactor;
    float y = (float)pyv * spResFactor;

    float Xd = (x - ppx) * Zval * fxInv;
    float Yd = (y - ppy) * Zval * fxInv;

    Xout[outIdx] = Xd;
    Yout[outIdx] = Yd;
    Zout[outIdx] = Zval;

    pxOut[outIdx] = pxv;
    pyOut[outIdx] = pyv;
    colOut[outIdx] = col;
}


//==================================================




// adjustToSystem Kernel
__global__ void adjustKernel(
    float* X, float* Y, float* Z,
    float xCenter, float yCenter, float zCenter,
    float xyScale,
    float xOffset, float yOffset, float zOffset,
    int N) // parameters already computed on CPU
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // same formula as CPU
    X[i] = (X[i] - xCenter) * xyScale + xOffset;
    Y[i] = (Y[i] - yCenter) * xyScale + yOffset;
    Z[i] = (Z[i] - zCenter) * xyScale + zOffset;
    
}




bool project2Dto3D(PointCloud& ptCloud,
                   const DatasetParameters& dataset,
                   const Config& config)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    size_t N = ptCloud.Z.size();
    std::cout << "[GPU] Point Cloud Size:" << N << "\n";
    if (N == 0) return false;

    float bgVal = ptCloud.getMaxZ();
    //float bgVal = 1000.0f;

    std::cout << "BGVAL value: " << bgVal << "\n";

    const int spResFactor = config.superResolutionFactor;
    const float fxInv = 1.0f / (dataset.CAM_FX_px * spResFactor);
    const float ppx = dataset.CAM_PX_px * spResFactor;
    const float ppy = dataset.CAM_PY_px * spResFactor;

    size_t fSize = N * sizeof(float);
    size_t u16Size = N * sizeof(uint16_t);
    size_t colSize = N * sizeof(RGB8);

    static_assert(sizeof(cv::Vec3b) == sizeof(RGB8), "Size mismatch");

    // =========================
    // STATIC GPU BUFFERS
    // =========================
    static float *d_Z = nullptr;
    static uint16_t *d_px = nullptr;
    static uint16_t *d_py = nullptr;
    static RGB8 *d_col = nullptr;

    static int *d_mask = nullptr;
    static int *d_scan = nullptr;

    static float *d_X = nullptr;
    static float *d_Y = nullptr;
    static float *d_Zout = nullptr;
    static uint16_t *d_pxOut = nullptr;
    static uint16_t *d_pyOut = nullptr;
    static RGB8 *d_colOut = nullptr;

    static void* d_tempStorage = nullptr;
    static size_t tempStorageBytes = 0;

    static size_t capacity = 0;

    if (N > capacity)
    {
        if (capacity > 0)
        {
            cudaFree(d_Z); cudaFree(d_px); cudaFree(d_py); cudaFree(d_col);
            cudaFree(d_mask); cudaFree(d_scan);
            cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Zout);
            cudaFree(d_pxOut); cudaFree(d_pyOut); cudaFree(d_colOut);
            cudaFree(d_tempStorage);
        }

        CUDA_CHECK(cudaMalloc(&d_Z, fSize));
        CUDA_CHECK(cudaMalloc(&d_px, u16Size));
        CUDA_CHECK(cudaMalloc(&d_py, u16Size));
        CUDA_CHECK(cudaMalloc(&d_col, colSize));

        CUDA_CHECK(cudaMalloc(&d_mask, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scan, N * sizeof(int)));

        CUDA_CHECK(cudaMalloc(&d_X, fSize));
        CUDA_CHECK(cudaMalloc(&d_Y, fSize));
        CUDA_CHECK(cudaMalloc(&d_Zout, fSize));
        CUDA_CHECK(cudaMalloc(&d_pxOut, u16Size));
        CUDA_CHECK(cudaMalloc(&d_pyOut, u16Size));
        CUDA_CHECK(cudaMalloc(&d_colOut, colSize));

        cub::DeviceScan::InclusiveSum(
            nullptr, tempStorageBytes,
            d_mask, d_scan, N);

        CUDA_CHECK(cudaMalloc(&d_tempStorage, tempStorageBytes));

        capacity = N;
    }

    // =========================
    // COPY INPUT
    // =========================
    CUDA_CHECK(cudaMemcpy(d_Z, ptCloud.Z.data(), fSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_px, ptCloud.px.data(), u16Size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, ptCloud.py.data(), u16Size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col,
        reinterpret_cast<RGB8*>(ptCloud.colors.data()),
        colSize,
        cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (N + block - 1) / block;

    // =========================
    // MASK
    // =========================
    computeMaskKernel<<<grid, block>>>(d_Z, d_mask, bgVal, N);

    // =========================
    // SCAN
    // =========================
    cub::DeviceScan::InclusiveSum(
        d_tempStorage, tempStorageBytes,
        d_mask, d_scan, N);

    int validCount;
    CUDA_CHECK(cudaMemcpy(&validCount, d_scan + (N - 1), sizeof(int), cudaMemcpyDeviceToHost));

    ptCloud.resize(validCount);

    // =========================
    // PROJECT + SCATTER
    // =========================
    projectScatterKernel<<<grid, block>>>(
        d_Z, d_px, d_py, d_col,
        d_X, d_Y, d_Zout,
        d_pxOut, d_pyOut, d_colOut,
        d_mask, d_scan,
        fxInv, ppx, ppy,
        spResFactor,
        N
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================
    // COPY BACK
    // =========================
    CUDA_CHECK(cudaMemcpy(ptCloud.X.data(), d_X, validCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ptCloud.Y.data(), d_Y, validCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ptCloud.Z.data(), d_Zout, validCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ptCloud.px.data(), d_pxOut, validCount * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ptCloud.py.data(), d_pyOut, validCount * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<RGB8*>(ptCloud.colors.data()),
        d_colOut,
        validCount * sizeof(RGB8),
        cudaMemcpyDeviceToHost));

    std::cout << "[GPU] Valid points: " << validCount << "\n";

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[GPU][TIME] project2Dto3D: " << ms << " ms\n";

    return true;
}




bool adjustToSystem(PointCloud& ptCloud,
                    const SystemSpec& spec,
                    const Config& config)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    size_t N = ptCloud.size();
    if (N == 0) return false;

    auto stats = ptCloud.computeStats();

    float xyScale = std::min(
        spec.display.width_mm / stats.xRange,
        spec.display.height_mm / stats.yRange
    );

    float xOffset = spec.display.width_mm / 2.f - stats.xCenter;
    float yOffset = spec.display.height_mm / 2.f - stats.yCenter;
    //float zOffset = -stats.zCenter;
    float cdp_mm = (spec.mla.displayDistance_mm * spec.mla.focalLength_mm) /
               (spec.mla.displayDistance_mm - spec.mla.focalLength_mm);

    float zOffset = 0.0f;

    switch (config.pointCloudMode) {
    case PointCloudMode::REAL:
        if (stats.zMin <= 0.f) zOffset = stats.zMin;
        break;

    case PointCloudMode::VIRTUAL:
        if (stats.zMax >= 0.f) zOffset = -stats.zMax;
        break;

    case PointCloudMode::MLA:
        zOffset = -stats.zCenter;
        break;

    case PointCloudMode::CDP:
        zOffset = cdp_mm;
        break;

    default:
        break;
    }

    float *d_X, *d_Y, *d_Z;

    CUDA_CHECK(cudaMalloc(&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, ptCloud.X.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, ptCloud.Y.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Z, ptCloud.Z.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (N + block - 1) / block;

    adjustKernel<<<grid, block>>>(
        d_X, d_Y, d_Z,
        stats.xCenter, stats.yCenter, stats.zCenter,
        xyScale,
        xOffset, yOffset, zOffset,
        N
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(ptCloud.X.data(), d_X, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ptCloud.Y.data(), d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ptCloud.Z.data(), d_Z, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Z);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[GPU][TIME] adjustToSystem: " << ms << " ms\n";

    return true;
}


} // namespace PointCloudGPU
} // namespace GPU



