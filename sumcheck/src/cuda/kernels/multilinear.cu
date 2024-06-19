#include "../includes/barretenberg/ecc/curves/bn254/fr.cuh"

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
