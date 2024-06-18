#include "../includes/barretenberg/ecc/curves/bn254/fr.cuh"

using namespace bb;

extern "C" __global__ void evaluate(fr* coeffs, fr* point, uint8_t num_vars, fr* monomial_evals) {
    fr coeff = coeffs[threadIdx.x].to_montgomery_form();
    if (coeff == fr::zero()) {
        monomial_evals[threadIdx.x] = fr::zero();
    } else {
        monomial_evals[threadIdx.x] = coeff;
        for (int i = 0; i < num_vars; i++) {
            if (((threadIdx.x >> i) & 1) == 1) {
                point[i].self_to_montgomery_form();
                monomial_evals[threadIdx.x] *= point[i];
            }
        }
        monomial_evals[threadIdx.x].self_from_montgomery_form();
    }
    return;
}
