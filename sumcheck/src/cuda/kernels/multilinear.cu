#include "../includes/barretenberg/ecc/curves/bn254/fr.cuh"

extern "C" __global__ void mul(bb::fr* elems, bb::fr* results) {
    elems[0].self_to_montgomery_form();
    elems[1].self_to_montgomery_form();
    bb::fr temp = elems[0] * elems[1];
    results[threadIdx.x] = temp.from_montgomery_form();
    return;
}
