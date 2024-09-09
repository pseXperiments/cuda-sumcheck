#include "../includes/barretenberg/ecc/curves/bn254/fr.cuh"
#include <stdint.h>
#include <stdio.h>
using namespace bb;

__device__ fr merge(fr* evals, fr x, const int start) {
    return x * (evals[start + 1] - evals[start]) + evals[start];
}

extern "C" __global__ void eval(fr* evals, fr* point, u_int32_t size, uint8_t offset, int num_blocks) {
    extern __shared__ fr evals_shared[];
    volatile int num_block_processed = 0;
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    evals_shared[tid] = evals[idx];
    __syncthreads();
    auto num_threads = size >> 1;
    auto i = offset;
    while (num_threads > 0) {
        if (tid < num_threads) {
            evals_shared[tid] = point[i] * (evals_shared[2 * tid + 1] - evals_shared[2 * tid]) + evals_shared[2 * tid];
        }
        __syncthreads();
        i++;
        num_threads >>= 1;
        __syncthreads();
    }
    if (tid == 0) {
        num_block_processed++;
        while (num_block_processed != num_blocks);
        evals[idx >> 10] = evals_shared[tid];
    }
}

extern "C" __global__ void convert_to_montgomery_form(fr* evals, u_int32_t size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    evals[idx].self_to_montgomery_form();
}
