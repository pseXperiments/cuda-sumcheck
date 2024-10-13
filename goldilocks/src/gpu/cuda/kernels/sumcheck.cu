#include "../includes/goldilocks/fp_impl.cuh"
#include "../includes/goldilocks/fp2_impl.cuh"

using namespace goldilocks;

extern "C" __global__ void sum(fp2* data, fp2* result, unsigned int stride, unsigned int index) {
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

extern "C" __global__ void eval_at_k_and_product(
    unsigned int num_vars, unsigned int initial_poly_size, unsigned int num_prods, fp2* polys, fp2* buf, fp* k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = 1 << (num_vars - 1);
    while (tid < stride) {
        if (num_prods == 1) {
            if (*k == fp::zero()) buf[tid] = polys[tid];
            else if (*k == fp::one()) buf[tid] = polys[tid + stride];
            else buf[tid] = (polys[tid + stride] - polys[tid]).scalar_mul(*k) + polys[tid];
        } else if (num_prods == 2) {
            if (*k == fp::zero()) buf[tid] = polys[tid] * polys[initial_poly_size + tid];
            else if (*k == fp::one()) buf[tid] = polys[tid + stride] * polys[initial_poly_size + tid + stride];
            else buf[tid] = polys[tid] * polys[initial_poly_size + tid] + (polys[tid + stride] * polys[initial_poly_size + tid + stride]).scalar_mul(fp(4))
                + (polys[tid] * polys[initial_poly_size + tid + stride]).scalar_mul(-fp(2)) + (polys[tid + stride] * polys[initial_poly_size + tid]).scalar_mul(-fp(2));
        } else if (num_prods == 3) {
            if (*k == fp::zero()) buf[tid] = polys[tid] * polys[initial_poly_size + tid] * polys[2 * initial_poly_size + tid];
            else if (*k == fp::one()) buf[tid] = polys[tid + stride] * polys[initial_poly_size + tid + stride] * polys[2 * initial_poly_size + tid + stride];
            else buf[tid] = ((polys[tid + stride] - polys[tid]).scalar_mul(*k) + polys[tid]) *
                ((polys[initial_poly_size + tid + stride] - polys[initial_poly_size + tid]).scalar_mul(*k) + polys[initial_poly_size + tid]) *
                ((polys[2 * initial_poly_size + tid + stride] - polys[2 * initial_poly_size + tid]).scalar_mul(*k) + polys[2 * initial_poly_size + tid]);
        }
        tid += blockDim.x * gridDim.x;
    }
}

extern "C" __global__ void fold_into_half_in_place(
    unsigned int num_vars, unsigned int initial_poly_size, unsigned int num_blocks_per_poly, fp2* polys, fp2* challenge
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
extern "C" __global__ void squeeze_challenge(fp2* challenges, unsigned int index) {
    if (threadIdx.x == 0) {
        challenges[index] = fp2(fp(1034));
    }
}
