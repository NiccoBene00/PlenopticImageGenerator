#include "gpu/PostProcessingGPU.cuh"
#include "data/SystemSpec.hpp"
#include "data/Config.hpp"

#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <algorithm>

#define MAX_SAMPLES 49

namespace GPU {
namespace PostProcessing {

// ======================
// UTILS FUNCTIONS
// ======================
__device__ int reflect(int p, int len) {
    if (p < 0) return -p;
    if (p >= len) return 2 * len - p - 2;
    return p;
}

__device__ int reflect101(int p, int len)
{
    if (len == 1) return 0;

    while (p < 0 || p >= len)
    {
        if (p < 0)
            p = -p - 1;
        else
            p = 2 * len - p - 1;
    }
    return p;
}

__device__ void sortArray(unsigned char* arr, int n) {
    for (int i = 1; i < n; ++i) {
        unsigned char key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

__device__ unsigned char medianSelect(unsigned char* arr, int n) {
    for (int i = 0; i <= n/2; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx])
                minIdx = j;
        }
        unsigned char tmp = arr[i];
        arr[i] = arr[minIdx];
        arr[minIdx] = tmp;
    }
    return arr[n/2];
}

// ======================
// KERNEL 
// ======================
__global__ void crackFilteringKernelROI(
    const uchar4* input,
    uchar4* output,
    int imgWidth,
    int imgHeight,
    const MicroimageGPU* microimages,
    int numMicro,
    int kernelRadius)
{
    int miIdx = blockIdx.z;
    if (miIdx >= numMicro) return;

    MicroimageGPU mi = microimages[miIdx];

    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;

    if (lx >= mi.width || ly >= mi.height) return;

    int x = mi.x + lx;
    int y = mi.y + ly;
    int idx = y * imgWidth + x;

    uchar4 center = input[idx];
    bool isCrack = (center.w == 0);

    unsigned char r[MAX_SAMPLES];
    unsigned char g[MAX_SAMPLES];
    unsigned char b[MAX_SAMPLES];
    unsigned char a[MAX_SAMPLES];

    int count = 0;

    for (int dy = -kernelRadius; dy <= kernelRadius; ++dy) {
        for (int dx = -kernelRadius; dx <= kernelRadius; ++dx) {

            int nx = min(max(mi.x + lx + dx, mi.x), mi.x + mi.width - 1);
            int ny = min(max(mi.y + ly + dy, mi.y), mi.y + mi.height - 1);

            uchar4 p = input[ny * imgWidth + nx];
    
            r[count] = p.x;
            g[count] = p.y;
            b[count] = p.z;
            a[count] = p.w;
            count++;
            
        }
    }

    if (count == 0) {
        output[idx] = center;
        return;
    }

    sortArray(r, count);
    sortArray(g, count);
    sortArray(b, count);
    //sortArray(a, count);

    int mid = count / 2;

    uchar4 out;
    /*
    out.x = medianSelect(r, count);
    out.y = medianSelect(g, count);
    out.z = medianSelect(b, count);
    out.w = medianSelect(a, count);
    */
    out.x = r[mid];
    out.y = g[mid];
    out.z = b[mid];
    out.w = center.w;
    

    output[idx] = isCrack ? out : center;
}

// ======================
// ROTATION KERNEL
// ======================
__global__ void rotateMicroimage180Kernel(
    const uchar4* input,
    uchar4* output,
    int width,
    int height,
    int microimageSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    int miX = (x / microimageSize) * microimageSize;
    int miY = (y / microimageSize) * microimageSize;

    int localX = x - miX;
    int localY = y - miY;

    int rotX = microimageSize - 1 - localX;
    int rotY = microimageSize - 1 - localY;

    int dstX = miX + rotX;
    int dstY = miY + rotY;

    if (dstX >= width || dstY >= height) return;

    output[dstY * width + dstX] = input[idx];
}

// ======================
// HOST FUNCTION
// ======================

// ROI-based filtering:
//   Filtering is constrained to microimage regions only, preventing
//   unwanted blending across neighboring lenslets
//
// Batch processing:
//   CUDA grid.z is limited to 65535.
//   Therefore, microimages are processed in batches when the number
//   of lenslets exceeds the maximum grid depth
//
// GPU design:
//   Each CUDA thread processes one pixel inside a microimage ROI
//
// Kernel radius:
//   The filtering radius is configurable through:
//
//          config.crackFilteringKernel

bool crackFiltering(
    cv::Mat& image,
    const std::vector<MicroimageGPU>& microimages,
    const Config& config)
{
    
    
    if (image.empty()) return false;

    if (!image.isContinuous())
        image = image.clone();

    if (image.type() != CV_8UC4) {
        std::cerr << "ERROR: image is not CV_8UC4\n";
        return false;
    }

    int width = image.cols;
    int height = image.rows;
    int numMicro = static_cast<int>(microimages.size());

    int kernelRadius = config.crackFilteringKernel / 2;

    size_t imgSize = width * height * sizeof(uchar4);
    size_t microSize = numMicro * sizeof(MicroimageGPU);

    uchar4* d_in = nullptr;
    uchar4* d_out = nullptr;
    MicroimageGPU* d_micro = nullptr;

    cudaMalloc(&d_in, imgSize);
    cudaMalloc(&d_out, imgSize);
    cudaMalloc(&d_micro, microSize);

    cudaMemcpy(d_in, image.ptr<uchar4>(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_micro, microimages.data(), microSize, cudaMemcpyHostToDevice);

    // initialize output
    cudaMemcpy(d_out, d_in, imgSize, cudaMemcpyDeviceToDevice);

    int maxW = 0, maxH = 0;
    for (const auto& m : microimages) {
        maxW = std::max(maxW, m.width);
        maxH = std::max(maxH, m.height);
    }

    dim3 block(16, 16);
    
    const int MAX_GRID_Z = 65535;

    for (int offset = 0; offset < numMicro; offset += MAX_GRID_Z)
    {
        int batch = std::min(MAX_GRID_Z, numMicro - offset);

        dim3 grid(
            (maxW + 15) / 16,
            (maxH + 15) / 16,
            batch
        );

        crackFilteringKernelROI<<<grid, block>>>(
            d_in,
            d_out,
            width,
            height,
            d_micro + offset,   
            batch,
            kernelRadius
        );
    }

    cudaDeviceSynchronize();

    // debug error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;

    cudaMemcpy(image.ptr<uchar4>(), d_out, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_micro);
    
    return true;
}

// ======================
// ROTATION HOST
// ======================


// Microimage size:
//   The microimage diameter is estimated from the physical MLA pitch
//   and display pixel size:
//
//          microimageSize = MLA pitch / display pixel size
//
// Parallelization:
//   Each CUDA thread processes one output pixel and computes its
//   corresponding rotated source coordinate.

bool rotateMicroimages(cv::Mat& image, const SystemSpec& spec)
{
    
    
    if (image.empty()) return false;

    if (!image.isContinuous())
        image = image.clone();

    if (image.type() != CV_8UC4) {
        std::cerr << "ERROR: image is not CV_8UC4\n";
        return false;
    }

    int width = image.cols;
    int height = image.rows;

    int microimageSize = static_cast<int>(
        std::round(spec.mla.pitch_mm / spec.display.pixelSize_mm)
    );

    size_t size = width * height * sizeof(uchar4);

    uchar4* d_in = nullptr;
    uchar4* d_out = nullptr;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, image.ptr<uchar4>(), size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    rotateMicroimage180Kernel<<<grid, block>>>(
        d_in, d_out,
        width, height,
        microimageSize
    );

    cudaDeviceSynchronize();

    cudaMemcpy(image.ptr<uchar4>(), d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    
    return true;
}

} // namespace PostProcessing
} // namespace GPU