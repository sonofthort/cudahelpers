#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace cudahelpers {
	template <class T>
	T *cmalloc(int n) {
		void *mem;
		
		if (cudaSuccess != cudaMalloc(&mem, sizeof(T) * n))
		{
			return NULL;
		}
		
		return (T*)mem;
	}

	int get1DBlockSize(int n, int maxThreadsPerBlock) {
		return n / maxThreadsPerBlock + (n % maxThreadsPerBlock ? 1 : 0);
	}

	int get1DThreadCount(int n, int maxThreadsPerBlock) {
		return n < maxThreadsPerBlock ? n : maxThreadsPerBlock;
	}
}
