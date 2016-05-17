#pragma once

#include <cuda.h>
#include <omp.h>

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
}
