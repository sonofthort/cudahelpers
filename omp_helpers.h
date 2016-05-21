#pragma once

#include <omp.h>

// A set of host-side parallel computation helpers which use OpenMP.

namespace cudahelpers {
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

	template <class DualVector, class F>
	bool parallel_commit(DualVector &v, F f) {
		parallel_for(v.size(), [f, &v](int i) {
			f(v[i], i);
		});

		return v.commit();
	}

	template <class DualVector, class F>
	bool parallel_commit(DualVector &v, int n, F f) {
		if (!v.resize(n)) {
			return false;
		}
		
		return parallel_commit(v, f);
	}
}
