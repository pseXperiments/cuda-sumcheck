#include "../includes/barretenberg/ecc/curves/bn254/fr.cuh"

using namespace bb;

extern "C" __global__ void mul(fr* elems, fr* results) {
    fr temp = elems[0] * elems[1];
    results[threadIdx.x] = temp.from_montgomery_form();
    return;
}
