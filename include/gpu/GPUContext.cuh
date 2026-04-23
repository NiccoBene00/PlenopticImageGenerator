#pragma once

#include <cstdint>

#include "gpu/GPUMemory.cuh"

namespace GPU {
	namespace Context {
		struct ContextPacked64 {
			uint8_t* d_output = nullptr;
			uint64_t* d_zBuffer = nullptr;

			~ContextPacked64() {
				GPU::Memory::freeDataMem(d_output);
				GPU::Memory::freeDataMem(d_zBuffer);
			}
		};

		struct ContextAtomic {
			uint8_t* d_output = nullptr;
			uint32_t* d_writeCount = nullptr;
			uint32_t* d_zBuffer = nullptr;

			~ContextAtomic() {
				GPU::Memory::freeDataMem(d_output);
				GPU::Memory::freeDataMem(d_zBuffer);
			}
		};
	}
}