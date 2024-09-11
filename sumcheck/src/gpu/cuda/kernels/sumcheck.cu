#include <stdint.h>
#include "../includes/barretenberg/ecc/curves/bn254/fr.cuh"
#include "../includes/transcript.cu"
#include "./multilinear.cu"

using namespace bb;

// TODO
__device__ fr combine_function(fr* evals, unsigned int start, unsigned int stride, unsigned int num_args) {
    fr result = fr::one();
    for (int i = 0; i < num_args; i++) result *= evals[start + i * stride];
    return result;
}

extern "C" __global__ void combine(fr* buf, unsigned int size, unsigned int num_args) {
    const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    while (idx < size) {
        buf[idx] = combine_function(buf, idx, size, num_args);
        idx += blockDim.x * gridDim.x;
    }
}

extern "C" __global__ void sum(
    fr* data, unsigned int stride, unsigned int index,
    uint8_t* start_transcript, uint8_t* cursor_transcript
) {
    Transcript t;
    t.init_transcript(start_transcript, cursor_transcript);
    const int tid = threadIdx.x;
    for (unsigned int s = stride; s > 0; s >>= 1) {
        int idx = tid;
        while (idx < s) {
            data[idx] += data[idx + s];
            idx += blockDim.x;
        }
        __syncthreads();
    }
    if (tid == 0) t.write_field_element(data[0]);
}

extern "C" __global__ void fold_into_half(
    unsigned int num_vars, unsigned int initial_poly_size, unsigned int num_blocks_per_poly, fr* polys, fr* buf,
    uint8_t* start_transcript, uint8_t* cursor_transcript
) {
    int tid = (blockIdx.x % num_blocks_per_poly) * blockDim.x + threadIdx.x;
    Transcript t;
    t.init_transcript(start_transcript, cursor_transcript);
    const int stride = 1 << (num_vars - 1);
    const int buf_offset = (blockIdx.x / num_blocks_per_poly) * stride;
    const int poly_offset = (blockIdx.x / num_blocks_per_poly) * initial_poly_size;
    fr challenge = t.squeeze_challenge();
    while (tid < stride) {
        if (challenge == fr::zero()) buf[buf_offset + tid] = polys[poly_offset + tid];
        else if (challenge == fr::one()) buf[buf_offset + tid] = polys[poly_offset + tid + stride];
        else buf[buf_offset + tid] = (challenge) * (polys[poly_offset + tid + stride] - polys[poly_offset + tid]) + polys[poly_offset + tid];
        tid += blockDim.x * num_blocks_per_poly;
    }
}

extern "C" __global__ void fold_into_half_in_place(
    unsigned int num_vars, unsigned int initial_poly_size, unsigned int num_blocks_per_poly, fr* polys,
    uint8_t* start_transcript, uint8_t* cursor_transcript
) {
    int tid = (blockIdx.x % num_blocks_per_poly) * blockDim.x + threadIdx.x;
    Transcript t;
    t.init_transcript(start_transcript, cursor_transcript);
    const int stride = 1 << (num_vars - 1);
    const int offset = (blockIdx.x / num_blocks_per_poly) * initial_poly_size;
    fr challenge = t.squeeze_challenge();
    while (tid < stride) {
        int idx = offset + tid;
        polys[idx] = (challenge) * (polys[idx + stride] - polys[idx]) + polys[idx];
        tid += blockDim.x * num_blocks_per_poly;
    }
}
