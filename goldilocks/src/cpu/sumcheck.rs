use std::ops::Mul;

use ff::PrimeField;
use goldilocks::ExtensionField;
use itertools::Itertools;

use crate::cpu::{arithmetic::barycentric_weights, parallel::parallelize};

use super::arithmetic::barycentric_interpolate;

pub(crate) fn eval_at_k_and_combine<E: ExtensionField + Mul<E::BaseField, Output = E>>(
    num_vars: usize,
    polys: &[&[E]],
    combine_function: &impl Fn(&Vec<E>) -> E,
    k: <E as ExtensionField>::BaseField,
) -> E {
    let evals = (0..1 << (num_vars - 1))
        .map(|idx| {
            let args = polys
                .iter()
                .map(|poly| (poly[idx + (1 << (num_vars - 1))] - poly[idx]) * k + poly[idx])
                .collect_vec();
            combine_function(&args)
        })
        .collect_vec();
    evals.into_iter().sum()
}

pub(crate) fn fold_into_half_in_place<E: ExtensionField>(poly: &mut [E], challenge: E) {
    let (poly0, poly1) = poly.split_at_mut(poly.len() >> 1);
    let poly1 = &*poly1;
    parallelize(poly0, |(poly0, start)| {
        poly0
            .iter_mut()
            .zip(poly1.iter().skip(start))
            .for_each(|(eval0, eval1)| {
                *eval0 = challenge * (*eval1 - *eval0) + *eval0;
            });
    });
}

pub(crate) fn verify_sumcheck<E: ExtensionField>(
    num_vars: usize,
    max_degree: usize,
    sum: E::BaseField,
    challenges: &[E],
    evals: &[&[E]],
) -> bool {
    let points_vec: Vec<E::BaseField> = (0..max_degree + 1)
        .map(|i| E::BaseField::from_u128(i as u128))
        .collect();
    let weights = barycentric_weights(&points_vec);
    // round 0
    let round_poly_eval_at_0 = evals[0][0];
    let round_poly_eval_at_1 = evals[0][1];
    let computed_sum = (round_poly_eval_at_0 + round_poly_eval_at_1).as_limbs()[0];
    if sum != computed_sum {
        return false;
    }
    let mut expected_sum =
        barycentric_interpolate::<E>(&weights, &points_vec, evals[0], &challenges[0]);
    // round 1..num_vars
    for round_index in 1..num_vars {
        if evals[round_index].len() != max_degree + 1 {
            return false;
        }
        let round_poly_eval_at_0 = evals[round_index][0];
        let round_poly_eval_at_1 = evals[round_index][1];
        let computed_sum = round_poly_eval_at_0 + round_poly_eval_at_1;

        // Check r_{i}(α_i) == r_{i+1}(0) + r_{i+1}(1)
        if computed_sum != expected_sum {
            println!("computed_sum : {:?}", computed_sum);
            println!("expected_sum : {:?}", expected_sum);
            println!("round index : {}", round_index);
            return false;
        }

        // Compute r_{i}(α_i) using barycentric interpolation
        expected_sum = barycentric_interpolate(
            &weights,
            &points_vec,
            evals[round_index],
            &challenges[round_index],
        );
    }
    true
}
