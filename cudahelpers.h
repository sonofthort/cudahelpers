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

	template <class F>
	void parallel_for(int begin, int end, F f) {
		#pragma omp parallel for
		for (int i = begin; i < end; ++i) {
			f(i);
		}
	}

	template <class F>
	void parallel_for(int end, F f) {
		return parallel_for(0, end, f);
	}
}
