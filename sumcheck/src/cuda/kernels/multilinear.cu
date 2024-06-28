#include "../includes/barretenberg/ecc/curves/bn254/fr.cuh"
#include <stdio.h>

using namespace bb;

extern "C" __global__ void evaluate(fr* coeffs, fr* point, uint8_t num_vars, fr* monomial_evals) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    fr coeff = coeffs[index].to_montgomery_form();
    if (coeff == fr::zero()) {
        monomial_evals[index] = fr::zero();
    } else {
        monomial_evals[index] = coeff;
        for (int i = 0; i < num_vars; i++) {
            if (((index >> i) & 1) == 1) {
                monomial_evals[index] *= point[i];
            }
        }
    }
    return;
}

extern "C" __global__ void evaluate_optimized(fr* coeffs, fr* point, uint8_t num_vars, fr* monomial_evals) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	auto step_size = 1;
	int number_of_threads = blockDim.x >> 1;
    bool evaluated = false;

	while (number_of_threads > 0)
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

		if (tid < number_of_threads) // still alive?
		{
			const auto fst = blockIdx.x * blockDim.x + tid * step_size * 2;
			const auto snd = fst + step_size;
			monomial_evals[fst] += monomial_evals[snd];
		}

		step_size <<= 1;
		number_of_threads >>= 1;
        __syncthreads();
	}
    if (tid == 0) {
        monomial_evals[idx].self_from_montgomery_form();
    }
}
