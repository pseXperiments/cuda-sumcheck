#include "../includes/barretenberg/ecc/curves/bn254/fr.cuh"

using namespace bb;

// TODO: Clookup combining equation
// h_function for range table
// evals[start]: f(x)
// evals[start + stride]..evals[start + table_dim * stride]: sigmas
// evals[start + (num_args - 1) * stride]: eq
// table equation: x_0 * 2^{table_dim-1} + x_1 * 2^{table_dim-2} + ...
__device__ fr combine_function(fr* evals, unsigned int start, unsigned int stride, unsigned int num_args, fr* gamma) {
    unsigned int table_dim = num_args - 2;
    fr result = fr::zero();
    for (int i = 1; i <= table_dim; i++)
        result += fr(1 << (i - 1)) * evals[start + i * stride];
    result = evals[start] - result;
    for (int i = 1; i <= table_dim; i++)
        result += gamma->pow(i) * evals[start + i * stride] * (evals[start + i * stride] - fr::one());
    result *= evals[start + (num_args - 1) * stride];
    return result;
}

extern "C" __global__ void combine(fr* buf, unsigned int size, unsigned int num_args, fr* gamma) {
    const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    while (idx < size) {
        buf[idx] = combine_function(buf, idx, size, num_args, gamma);
        idx += blockDim.x * gridDim.x;
    }
}

extern "C" __global__ void sum(fr* data, fr* result, unsigned int stride, unsigned int index) {
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
    unsigned int num_vars, unsigned int initial_poly_size, unsigned int num_blocks_per_poly, fr* polys, fr* buf, const fr* eval_point
) {
    int tid = (blockIdx.x % num_blocks_per_poly) * blockDim.x + threadIdx.x;
    const int stride = 1 << (num_vars - 1);
    const int buf_offset = (blockIdx.x / num_blocks_per_poly) * stride;
    const int poly_offset = (blockIdx.x / num_blocks_per_poly) * initial_poly_size;
    while (tid < stride) {
        if (*eval_point == fr::zero()) buf[buf_offset + tid] = polys[poly_offset + tid];
        else if (*eval_point == fr::one()) buf[buf_offset + tid] = polys[poly_offset + tid + stride];
        else buf[buf_offset + tid] = (*eval_point) * (polys[poly_offset + tid + stride] - polys[poly_offset + tid]) + polys[poly_offset + tid];
        tid += blockDim.x * num_blocks_per_poly;
    }
}

extern "C" __global__ void fold_into_half_in_place(
    unsigned int num_vars, unsigned int initial_poly_size, unsigned int num_blocks_per_poly, fr* polys, const fr* challenge
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
