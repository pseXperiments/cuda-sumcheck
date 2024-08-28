#include "../includes/barretenberg/ecc/curves/bn254/fr.cuh"
#include "./multilinear.cu"

using namespace bb;

__device__ void sum(fr* data, const int stride) {
    const int tid = threadIdx.x;
    for (unsigned int s = stride; s > 0; s >>= 1) {
        int idx = tid;
        while (idx < s) {
            data[idx] += data[idx + s];
            idx += blockDim.x;
        }
        __syncthreads();
    }
}

// TODO
__device__ fr combine_function(fr* evals, unsigned int start, unsigned int stride, unsigned int num_args) {
    fr result = fr::zero();
    for (int i = 0; i < num_args; i++) result += evals[start + i * stride];
    return result;
}

extern "C" __global__ void combine_and_sum(
    fr* buf, fr* result, unsigned int size, unsigned int num_args, unsigned int index
) {
    const int tid = threadIdx.x;
    int idx = tid;
    while (idx < size) {
        buf[idx] = combine_function(buf, idx, size, num_args);
        idx += blockDim.x;
    }
    __syncthreads();
    sum(buf, size >> 1);
    if (tid == 0) result[index] = buf[0];
}

extern "C" __global__ void fold_into_half(
    unsigned int num_vars, unsigned int initial_poly_size, unsigned int num_blocks_per_poly, fr* polys, fr* buf, fr* challenge
) {
    int tid = (blockIdx.x % num_blocks_per_poly) * blockDim.x + threadIdx.x;
    const int stride = 1 << (num_vars - 1);
    const int buf_offset = (blockIdx.x / num_blocks_per_poly) * stride;
    const int poly_offset = (blockIdx.x / num_blocks_per_poly) * initial_poly_size;
    while (tid < stride) {
        buf[buf_offset + tid] = (*challenge) * (polys[poly_offset + tid + stride] - polys[poly_offset + tid]) + polys[poly_offset + tid];
        tid += blockDim.x * num_blocks_per_poly;
    }
}

extern "C" __global__ void fold_into_half_in_place(
    unsigned int num_vars, unsigned int initial_poly_size, unsigned int num_blocks_per_poly, fr* polys, fr* challenge
) {
    int tid = (blockIdx.x % num_blocks_per_poly) * blockDim.x + threadIdx.x;
    const int stride = 1 << (num_vars - 1);
    const int offset = (blockIdx.x / num_blocks_per_poly) * initial_poly_size;
    while (tid < stride) {
        int idx = offset + tid;
        polys[idx] = (*challenge) * (polys[idx + stride] - polys[idx]) + polys[idx];
        tid += blockDim.x * num_blocks_per_poly;
    }
}

// TODO : Pass transcript and squeeze random challenge using hash function
extern "C" __global__ void squeeze_challenge(fr* challenges, unsigned int index) {
    if (threadIdx.x == 0) {
        challenges[index] = fr(1034);
    }
}
