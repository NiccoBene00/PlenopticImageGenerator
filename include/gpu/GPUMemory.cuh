#pragma once

#include <cuda_runtime.h>
#include <string>
#include <iostream>

#include "gpu/GPUUtils.cuh"

namespace GPU {
    namespace Memory {
        // ===============================
        // KERNEL DEFINITION (in header)
        // ===============================
        template <typename T>
        __global__ void initArrayKernel(T* d_array, T value, size_t count)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < count) {
                d_array[idx] = value;
            }
        }

        // ===============================
        // LAUNCHER DEFINITION (in header)
        // ===============================
        template <typename T>
        inline bool initGlobalMemArray(T* d_array, T value, size_t count, std::string& errorMsg)
        {
            if (!d_array || count == 0)
                return true;
            constexpr int threadsPerBlock = 256;
            int blocksPerGrid = static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock);
            initArrayKernel<T> << <blocksPerGrid, threadsPerBlock >> > (d_array, value, count);
            return GPU::Utils::CUDA_CHECK_ERR(cudaDeviceSynchronize(), errorMsg);
        }

        // ===============================
        // INLINE HELPERS (already correct)
        // ===============================
        template <typename T>
        inline bool allocGlobalMem(T*& d_ptr, size_t count, std::string& errorMsg)
        {
            d_ptr = nullptr;
            return GPU::Utils::CUDA_CHECK_ERR(
                cudaMalloc(&d_ptr, count * sizeof(T)), errorMsg);
        }

        template <typename T>
        inline bool sendToGlobalMem(T*& d_ptr, const T* hostData, size_t count, std::string& errorMsg)
        {
            d_ptr = nullptr;
            size_t size = count * sizeof(T);
            if (!GPU::Utils::CUDA_CHECK_ERR(cudaMalloc(&d_ptr, size), errorMsg))
                return false;
            if (!GPU::Utils::CUDA_CHECK_ERR(
                cudaMemcpy(d_ptr, hostData, size, cudaMemcpyHostToDevice),
                errorMsg))
            {
                cudaFree(d_ptr);
                d_ptr = nullptr;
                return false;
            }
            return true;
        }

        template <typename T>
        inline bool sendToConstantMem(const T& hostValue, const T& symbol, std::string& errorMsg, size_t count = 1)
        {
            return GPU::Utils::CUDA_CHECK_ERR(
                cudaMemcpyToSymbol(symbol, &hostValue, count * sizeof(T)),
                errorMsg);
        }

        template <typename T>
        inline bool getFromGlobalMem(T* hostPtr, const T* devicePtr, size_t count, std::string& errorMsg)
        {
            return GPU::Utils::CUDA_CHECK_ERR(
                cudaMemcpy(hostPtr, devicePtr, count * sizeof(T), cudaMemcpyDeviceToHost),
                errorMsg);
        }

        template <typename T>
        inline void freeDataMem(T*& d_ptr) noexcept
        {
            if (d_ptr) {
                cudaError_t err = cudaFree(d_ptr);
                if (err != cudaSuccess) {
                    std::cerr << "cudaFree failed: "
                        << cudaGetErrorString(err) << std::endl;
                }
                d_ptr = nullptr;
            }
        }
    }
}