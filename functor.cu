#pragma once

#include "cudahelpers.h"

#define CUDAHELPERS_FUNC __host__ __device__

namespace cudahelpers
{
	template <class T>
	struct Ref {
		CUDAHELPERS_FUNC Ref(T &value) :
			value(&value)
		{}
		
		CUDAHELPERS_FUNC operator const T &() const {
			return *value;
		}
		
		CUDAHELPERS_FUNC operator T &() {
			return *value;
		}
		
		CUDAHELPERS_FUNC const T &get() const {
			return *value;
		}
		
		CUDAHELPERS_FUNC T &get() {
			return *value;
		}
		
	private:
		T *value;
	};
	
	template <class T>
	CUDAHELPERS_FUNC  Ref<T> ref(T &value) {
		return Ref<T>(value);
	}
	
	template <class T>
	CUDAHELPERS_FUNC Ref<const T> ref(const T &value) {
		return Ref<const T>(value);
	}
	
	template <class T, class = void>
	struct ArrayView {
		CUDAHELPERS_FUNC ArrayView() :
			m_data(NULL),
			m_size(0)
		{}

		CUDAHELPERS_FUNC ArrayView(T *data, int size) :
			m_data(data),
			m_size(size)
		{}

		CUDAHELPERS_FUNC int size() const {
			return m_size;
		}

		CUDAHELPERS_FUNC T *data() const {
			return m_data;
		}

		CUDAHELPERS_FUNC T &operator[](int i) const {
			return m_data[i];
		}

		CUDAHELPERS_FUNC ArrayView split(int begin) const {
			return split(begin, m_size - begin);
		}

		CUDAHELPERS_FUNC ArrayView split(int begin, int end) const {
			return ArrayView(m_data + begin, end - begin);
		}

		CUDAHELPERS_FUNC void set(T *data, int size) {
			m_data = data;
			m_size = size;
		}

		CUDAHELPERS_FUNC void clear() {
			m_data = NULL;
			m_size = 0;
		}

	private:
		T *m_data;
		int m_size;
	};
	
	template <class A, class B>
	struct CombineFuncs : private A {
		CUDAHELPERS_FUNC CombineFuncs(const A &a, const B &b) :
			A(a),
			b(b)
		{}
		
		CUDAHELPERS_FUNC void operator()() const {
			static_cast<A&>(*this)();
			b();
		}
		
		template <class A1>
		CUDAHELPERS_FUNC void operator()(A1 a1) const {
			static_cast<A&>(*this)(a1);
			b(a1);
		}
		
		template <class A1, class A2>
		CUDAHELPERS_FUNC void operator()(A1 a1, A2 a2) const {
			static_cast<A&>(*this)(a1, a2);
			b(a1, a2);
		}
		
		template <class A1, class A2, class A3>
		CUDAHELPERS_FUNC void operator()(A1 a1, A2 a2, A3 a3) const {
			static_cast<A&>(*this)(a1, a2, a3);
			b(a1, a2, a3);
		}
		
	private:
		B b;
	};
	
	template <class A, class B>
	CombineFuncs<A, B> combine_funcs(const A &a, const B &b) {
		return CombineFuncs<A, B>(a, b);
	}
	
	template <class A, class B, class C>
	CombineFuncs<CombineFuncs<A, B>, C> combine_funcs(const A &a, const B &b, const C &c) {
		return combine_funcs(combine_funcs(a, b), c);
	}

	template <class Func>
	__global__ void kernel_for(int end, Func func) {
		const int index = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (index < end) {
			func(index);
		}
	}
	
	template <class T, class Func>
	__global__ void kernel_for(int begin, int end, Func func)
	{
		const int index = blockIdx.x * blockDim.x + threadIdx.x + begin;
		
		if (index < end) {
			func(index);
		}
	}

	template <class T, class Func>
	__global__ void kernel_for_each(T *arr, int size, Func func) {
		const int index = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (index < size) {
			func(arr[index]);
		}
	}

	template <class T, class Func>
	__global__ void kernel_for_each(const T *arr, int size, Func func) {
		const int index = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (index < size) {
			func(arr[index]);
		}
	}

	template <class T, class Func>
	__global__ void kernel_iv(T *data, int size, Func func) {
		const int index = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (index < size) {
			func(index, ref(data[index]));
		}
	}

	template <class Func>
	void for_n(int end, int maxThreadsPerBlock, Func func) {
		kernel_for<<<
			cudahelpers::get1DBlockSize(end, maxThreadsPerBlock),
			cudahelpers::get1DThreadCount(end, maxThreadsPerBlock)
		>>>(end, func);
	}

	template <class Func>
	void for_n(int begin, int end, int maxThreadsPerBlock, Func func) {
		const int n = end - begin;

		kernel_for<<<
			cudahelpers::get1DBlockSize(n, maxThreadsPerBlock),
			cudahelpers::get1DThreadCount(n, maxThreadsPerBlock)
		>>>(begin, end, func);
	}

	template <class T, class Func>
	void for_each(T *arr, int size, int maxThreadsPerBlock, Func func) {
		kernel_for_each<<<
			cudahelpers::get1DBlockSize(size, maxThreadsPerBlock),
			cudahelpers::get1DThreadCount(size, maxThreadsPerBlock)
		>>>(arr, size, func);
	}

	template <class T, class Func>
	void for_each(const T *arr, int size, int maxThreadsPerBlock, Func func) {
		kernel_for_each<<<
			cudahelpers::get1DBlockSize(end, maxThreadsPerBlock),
			cudahelpers::get1DThreadCount(end, maxThreadsPerBlock)
		>>>(arr, size, func);
	}
	
	template <class T, class Func>
	void iv(T *data, int size, int maxThreadsPerBlock, Func func) {
		kernel_iv<<<
			cudahelpers::get1DBlockSize(size, maxThreadsPerBlock),
			cudahelpers::get1DThreadCount(size, maxThreadsPerBlock)
		>>>(data, size, func);
	}
}
