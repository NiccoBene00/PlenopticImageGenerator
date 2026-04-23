/*
#include "gpu/PostProcessingGPU.cuh"
#include "data/SystemSpec.hpp"
#include "data/Config.hpp"


#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define MAX_SAMPLES 49

namespace GPU {
namespace PostProcessing {



__device__ inline int checkborder(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

// insertion sort (used for median filtering)
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


__device__ int reflect(int p, int len) {
    if (p < 0) return -p;
    if (p >= len) return 2 * len - p - 2;
    return p;
}


//Crack Filtering Kernel
//Logic: one cuda thread manipulates one bit
__global__ void crackFilteringMedianKernel(
    const uchar4* input,
    uchar4* output,
    int width,
    int height,
    int microimageSize,
    int kernelRadius)
{
    //thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //check if Im out of border
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uchar4 center = input[idx];

    // if the pixel is not crack then simple copy
    // (like in the cpu version)
    
    //if (center.w != 0) {
    //    output[idx] = center;
    //    return;
    //}
    bool isCrack = (center.w == 0);

    // microimage origin
    int miX = (x / microimageSize) * microimageSize;
    int miY = (y / microimageSize) * microimageSize;

    int kSize = 2 * kernelRadius + 1;
    //int maxSamples = 49;
    unsigned char r[MAX_SAMPLES];
    unsigned char g[MAX_SAMPLES];
    unsigned char b[MAX_SAMPLES];
    unsigned char a[MAX_SAMPLES];

    int count = 0;


    for (int dy = -kernelRadius; dy <= kernelRadius; ++dy) {
        for (int dx = -kernelRadius; dx <= kernelRadius; ++dx) {

            //int nx = x + dx;
            //int ny = y + dy;
            
            //avoid reading pixel outside the total img
            //nx = checkborder(nx, 0, width - 1);
            //ny = checkborder(ny, 0, height - 1);

            // avoid reading pixel outside the microimg
            //nx = checkborder(nx, miX, miX + microimageSize - 1);
            //ny = checkborder(ny, miY, miY + microimageSize - 1);

            // local coordinates inside microimage
            int localX = (x - miX) + dx;
            int localY = (y - miY) + dy;

            
            
            // reflect INSIDE microimage
            localX = reflect(localX, microimageSize);
            localY = reflect(localY, microimageSize);

            // back to global coordinates
            int nx = miX + localX;
            int ny = miY + localY;

            uchar4 p = input[ny * width + nx];

            r[count] = p.x;
            g[count] = p.y;
            b[count] = p.z;
            a[count] = p.w;

            count++;
        }
    }

    // median filter (same of cv::medianBlur in the cpu version)
    sortArray(r, count);
    sortArray(g, count);
    sortArray(b, count);
    sortArray(a, count);

    int mid = count / 2;

    uchar4 out;
    out.x = r[mid];
    out.y = g[mid];
    out.z = b[mid];
    out.w = a[mid];

    //output[idx] = out;
    if (isCrack)
        output[idx] = out;
    else
        output[idx] = center;
}



//kernel rotation for each microimage
//Logic: each thread manipulates one pixel
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
    
    //topìleft corner of each microimg
    int miX = (x / microimageSize) * microimageSize;
    int miY = (y / microimageSize) * microimageSize;
    
    //local coordinates inside the microimg
    int localX = x - miX;
    int localY = y - miY;
    
    //rotation of 180 
    int rotX = microimageSize - 1 - localX;
    int rotY = microimageSize - 1 - localY;
    
    //back to gloabal coordinates
    int dstX = miX + rotX;
    int dstY = miY + rotY;

    if (dstX >= width || dstY >= height) return;

    output[dstY * width + dstX] = input[idx];
}


bool crackFiltering(cv::Mat& image, const SystemSpec& spec, const Config& config)
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

    int kernelRadius = config.crackFilteringKernel / 2;
    
    //memory dimension
    size_t size = width * height * sizeof(uchar4);

    uchar4* d_in = nullptr;
    uchar4* d_out = nullptr;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, image.ptr<uchar4>(), size, cudaMemcpyHostToDevice);

    dim3 block(16, 16); //in such way i set 256 thread per block
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    crackFilteringMedianKernel<<<grid, block>>>(
        d_in, d_out,
        width, height,
        microimageSize,
        kernelRadius
    );

    cudaDeviceSynchronize();

    cudaMemcpy(image.ptr<uchar4>(), d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    //maybe cuda errors checking miss here

    return true;
}



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

    //todo: check for cuda errors

    return true;
}

} // namespace PostProcessing
} // namespace GPU

*/

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

            //int nx_l = reflect101(lx + dx, mi.width);
            //int ny_l = reflect101(ly + dy, mi.height);

            //int nx = mi.x + nx_l;
            //int ny = mi.y + ny_l;

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

    // inizializza output
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