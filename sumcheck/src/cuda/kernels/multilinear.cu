#include "../includes/barretenberg/ecc/curves/bn254/fr.cuh"
#include <stdio.h>

using namespace bb;

__device__ fr merge(fr* evals, fr* point, uint8_t point_index, u_int32_t chunk_size) {
    const int start = chunk_size * (blockIdx.x * blockDim.x + threadIdx.x);
    auto step = 2;
    while (chunk_size > 1) {
        for (int i = 0; i < chunk_size / 2; i++) {
            auto fst = start + step * i;
            auto snd = fst + step / 2;
            evals[fst] = point[point_index] * (evals[snd] - evals[fst]) + evals[fst];
        }
        chunk_size >>= 1;
        step <<= 1;
        point_index++;
    }
    return evals[start];
}

extern "C" __global__ void eval(fr* evals, fr* buf, fr* point, u_int32_t size, u_int32_t chunk_size, uint8_t offset) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t log2_chunk_size = log2(chunk_size);
    u_int32_t num_threads = ceil(size / chunk_size);
    auto i = offset;
    while (num_threads > 0) {
        if (tid < num_threads) {
            buf[idx] = merge(evals, point, i, chunk_size);
        }
        __syncthreads();
        if (tid == 0) {
            memcpy(&evals[chunk_size * blockIdx.x * blockDim.x], &buf[blockIdx.x * blockDim.x], num_threads * 32);
        }
        i += log2_chunk_size;
        num_threads >>= log2_chunk_size;
        __syncthreads();
    }
}

extern "C" __global__ void convert_to_montgomery(fr* evals, u_int32_t size, u_int32_t chunk_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < chunk_size; i++) {
        evals[chunk_size * idx + i].self_to_montgomery_form();
    }
}
