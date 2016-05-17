#pragma once

#include "cudahelpers.h"

// DualVector manages a device array from the perspective of a local host copy.

// Data is viewed and updated through the host copy, and committed to the device with commit().
// Data can be retrieved from the device with update().

// commit() has some overloads for setting values and committing in one step, which may be a common pattern.

// Please note that updating a device array is typically much faster with kernels.
// This class provides a simple interface at the cost of some performance.
// Best to use for initializations only, or for unoccasional reinitializations.
// Also useful for small arrays, which come into use as kernal arguments.

namespace cudahelpers {
	namespace detail {
		template <class T, class Alloc, class Free>
		struct GenericBuffer {
			GenericBuffer() :
				m_data(NULL),
				m_size(0)
			{}

			GenericBuffer(T *data, int n) :
				m_data(data),
				m_size(n)
			{}

			GenericBuffer(int n) :
				m_data(NULL),
				m_size(0)
			{
				resize(n);
			}

			~GenericBuffer() {
				if (m_data) {
					Free()(m_data);
				}
			}

			T *data() {
				return m_data;
			}

			const T *data() const {
				return m_data;
			}

			int size() const {
				return m_size;
			}

			void clear() {
				if (m_data) {
					Free()(m_data);
					m_data = NULL;
					m_size = 0;
				}
			}

			bool resize(int n) {
				if (n <= m_size) {
					return true;
				}

				T * const d = (T*)Alloc()(sizeof(T) * n);

				if (d) {
					if (m_data) {
						Free()(m_data);
					}

					m_data = d;
					m_size = n;

					return true;
				}

				return false;
			}

		private:
			T *m_data;
			int m_size;
		};

		struct HostAlloc {
			void *operator()(size_t n) const {
				return malloc(n);
			}
		};

		struct HostFree {
			void operator()(void *data) const {
				free(data);
			}
		};

		struct DeviceAlloc {
			void *operator()(size_t n) const {
				void *result;

				if (cudaSuccess != cudaMalloc(&result, n))
				{
					return NULL;
				}

				return result;
			}
		};

		struct DeviceFree {
			void operator()(void *data) const {
				cudaFree(data);
			}
		};
	}

	template <class T>
	class DualVector {
		typedef detail::GenericBuffer<T, detail::HostAlloc, detail::HostFree> HostBuffer;
		typedef detail::GenericBuffer<T, detail::DeviceAlloc, detail::DeviceFree> DeviceBuffer;

		DualVector(const DualVector &v);

	public:
		DualVector() :
			m_size(0)
		{}

		T *data() {
			return m_hostBuff.data();
		}

		const T *data() const {
			return m_hostBuff.data();
		}

		T *device_data() {
			return m_deviceBuff.data();
		}

		const T *device_data() const {
			return m_deviceBuff.data();
		}

		int size() const {
			return m_size;
		}

		void clear() {
			m_deviceBuff.clear();
			m_hostBuff.clear();
			m_size = 0;
		}

		bool reserve(int n) {
			return m_deviceBuff.resize(n) && m_hostBuff.resize(n);
		}

		bool resize(int n, const T &v = T()) {
			if (!reserve(n)) {
				return false;
			}

			const int currentSize = m_size;
			T * const data = m_hostBuff.data();

			for (int i = currentSize; i < n; ++i) {
				data[i] = v;
			}

			m_size = n;

			return true;
		}

		bool commit() {
			if (!m_deviceBuff.resize(m_size)) {
				return false;
			}

			return cudaSuccess == cudaMemcpy(m_deviceBuff.data(), m_hostBuff.data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
		}

		bool update() {
			if (!m_hostBuff.resize(m_size)) {
				return false;
			}

			return cudaSuccess == cudaMemcpy(m_deviceBuff.data(), m_hostBuff.data(), m_size * sizeof(T), cudaMemcpyDeviceToHost);
		}

		T *begin() {
			return m_hostBuff.data();
		}

		T *end() {
			return m_hostBuff.data() + m_size;
		}

		const T *begin() const {
			return m_hostBuff.data();
		}

		const T *end() const {
			return m_hostBuff.data() + m_size;
		}

		T &operator[](int i) {
			return m_hostBuff.data()[i];
		}

		const T &operator[](int i) const {
			return m_hostBuff.data()[i];
		}

		bool push_back(const T &v) {
			if (m_hostBuff.size() == m_size) {
				if (!m_hostBuff.resize(m_size == 0 ? 8 : m_size * 2)) {
					return false;
				}
			}

			m_hostBuff.data()[m_size] = v;

			++m_size;

			return true;
		}

		T &front() {
			return m_hostBuff.data()[0];
		}

		const T &front() const {
			return m_hostBuff.data()[0];
		}

		T &back() {
			return m_hostBuff.data()[m_size - 1];
		}

		const T &back() const {
			return m_hostBuff.data()[m_size - 1];
		}

		template <class F>
		void parallel_for_each(F f) {
			const int n = m_size;
			T * const buff = m_hostBuff.data();

			#pragma omp parallel for
			for (int i = 0; i < n; ++i) {
				f(buff[i], i);
			}
		}

		template <class F>
		void parallel_for_each(F f) const {
			const int n = m_size;
			const T * const buff = m_hostBuff.data();

			#pragma omp parallel for
			for (int i = 0; i < n; ++i) {
				f(buff[i], i);
			}
		}

		template <class F>
		bool commit(F f) {
			parallel_for_each(f);

			return commit();
		}

		template <class F>
		bool commit(int n, F f) {
			if (!resize(n)) {
				return false;
			}

			parallel_for_each(f);

			return commit();
		}

	private:
		DeviceBuffer m_deviceBuff;
		HostBuffer m_hostBuff;
		int m_size;
	};
}
