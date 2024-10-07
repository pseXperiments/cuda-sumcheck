#include "../includes/goldilocks/fp_impl.cuh"

using namespace goldilocks;

// TODO
__device__ fp combine_function(fp* evals, unsigned int start, unsigned int stride, unsigned int num_args) {
    fp result = fp::one();
    for (int i = 0; i < num_args; i++) result *= evals[start + i * stride];
    return result;
}

extern "C" __global__ void combine(fp* buf, unsigned int size, unsigned int num_args) {
    const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    while (idx < size) {
        buf[idx] = combine_function(buf, idx, size, num_args);
        idx += blockDim.x * gridDim.x;
    }
}

extern "C" __global__ void sum(fp* data, fp* result, unsigned int stride, unsigned int index) {
    const int tid = threadIdx.x;
    for (unsigned int s = stride; s > 0; s >>= 1) {
        int idx = tid;
        while (idx < s) {
            data[idx] += data[idx + s];
            idx += blockDim.x;
        }
        __syncthreads();
    }
    if (tid == 0) result[index] = data[0];
}

extern "C" __global__ void fold_into_half(
    unsigned int num_vars, unsigned int initial_poly_size, unsigned int num_blocks_per_poly, fp* polys, fp* buf, fp* challenge
) {
    int tid = (blockIdx.x % num_blocks_per_poly) * blockDim.x + threadIdx.x;
    const int stride = 1 << (num_vars - 1);
    const int buf_offset = (blockIdx.x / num_blocks_per_poly) * stride;
    const int poly_offset = (blockIdx.x / num_blocks_per_poly) * initial_poly_size;
    while (tid < stride) {
        if (*challenge == fp::zero()) buf[buf_offset + tid] = polys[poly_offset + tid];
        else if (*challenge == fp::one()) buf[buf_offset + tid] = polys[poly_offset + tid + stride];
        else buf[buf_offset + tid] = (*challenge) * (polys[poly_offset + tid + stride] - polys[poly_offset + tid]) + polys[poly_offset + tid];
        tid += blockDim.x * num_blocks_per_poly;
    }
}

extern "C" __global__ void fold_into_half_in_place(
    unsigned int num_vars, unsigned int initial_poly_size, unsigned int num_blocks_per_poly, fp* polys, fp* challenge
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
extern "C" __global__ void squeeze_challenge(fp* challenges, unsigned int index) {
    if (threadIdx.x == 0) {
        challenges[index] = fp(1034);
    }
}
