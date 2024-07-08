#include "../includes/barretenberg/ecc/curves/bn254/fr.cuh"
#include <stdio.h>

using namespace bb;

extern "C" __global__ void eval_by_coeff(fr* coeffs, fr* point, uint8_t num_vars, fr* monomial_evals) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	auto step_size = 1;
	int num_threads = blockDim.x >> 1;
    bool evaluated = false;

	while (num_threads > 0)
	{
        if (!evaluated) {
            fr coeff = coeffs[idx].to_montgomery_form();
            monomial_evals[idx] = coeff;
            for (int i = 0; i < num_vars; i++) {
                monomial_evals[idx] *= (((idx >> i) & 1) ? point[i] : fr::one());
            }
            evaluated = true;
            __syncthreads();
            continue;
        }

		if (tid < num_threads) // still alive?
		{
			const auto fst = blockIdx.x * blockDim.x + tid * step_size * 2;
			const auto snd = fst + step_size;
			monomial_evals[fst] += monomial_evals[snd];
		}

		step_size <<= 1;
		num_threads >>= 1;
        __syncthreads();
	}
    if (tid == 0) {
        monomial_evals[idx].self_from_montgomery_form();
    }
}

extern "C" __global__ void eval(fr* evals, fr* point, uint8_t num_vars, fr* buf) {
    const int tid = threadIdx.x;
    auto i = 0;
    auto num_threads = 1 << (num_vars - 1);
    while (num_threads > 0) {
        if (tid < num_threads) {
            buf[tid] = point[i] * (evals[2 * tid + 1] - evals[2 * tid]) + evals[2 * tid];
        }
        __syncthreads();
        if (tid == 0) {
            memcpy(evals, buf, num_threads * 32);
        }
        i++;
        num_threads >>= 1;
        __syncthreads();
    }
    if (tid == 0) {
        buf[0].self_from_montgomery_form();
    }
}
