#pragma once

#include "cudahelpers.h"

namespace cudahelpers
{
	template <class Func>
	void for_n(int end, int maxThreadsPerBlock, Func func);

	template <class Func>
	void for_n(int begin, int end, int maxThreadsPerBlock, Func func);

	template <class T, class Func>
	void for_each(T *arr, int size, int maxThreadsPerBlock, Func func);

	template <class T, class Func>
	void for_each(const T *arr, int size, int maxThreadsPerBlock, Func func);
	
	template <class T, class Func>
	void iv(T *data, int size, int maxThreadsPerBlock, Func func);	
}