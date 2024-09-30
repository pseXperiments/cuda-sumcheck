use std::io::Cursor;

use ff::PrimeField;
use itertools::Itertools;

use crate::cpu::{arithmetic::barycentric_weights, parallel::parallelize};

use super::super::transcript::*;
use super::arithmetic::barycentric_interpolate;

pub(crate) fn eval_at_k_and_combine<F: PrimeField>(
    num_vars: usize,
    polys: &[&[F]],
    combine_function: &impl Fn(&Vec<F>) -> F,
    k: F,
) -> F {
    let evals = (0..1 << (num_vars - 1))
        .map(|idx| {
            let args = polys
                .iter()
                .map(|poly| k * (poly[idx + (1 << (num_vars - 1))] - poly[idx]) + poly[idx])
                .collect_vec();
            combine_function(&args)
        })
        .collect_vec();
    evals.into_iter().sum()
}

pub(crate) fn fold_into_half_in_place<F: PrimeField>(poly: &mut [F], challenge: F) {
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

pub(crate) fn verify_sumcheck<F: PrimeField>(
    num_vars: usize,
    max_degree: usize,
    sum: F,
    challenges: &[F],
    evals: &[&[F]],
) -> bool {
    let points_vec: Vec<F> = (0..max_degree + 1)
        .map(|i| F::from_u128(i as u128))
        .collect();
    let weights = barycentric_weights(&points_vec);
    let mut expected_sum = sum;
    for round_index in 0..num_vars {
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

pub(crate) fn verify_sumcheck_transcript<F: PrimeField + halo2curves::serde::SerdeObject>(
    num_vars: usize,
    max_degree: usize,
    sum: F,
    transcript: Vec<u8>,
) -> bool {
    let stream = Cursor::new(transcript);
    let mut transcript = CudaKeccakTranscript {
        stream,
        state: F::ZERO,
        inner: None,
    };
    let points_vec: Vec<F> = (0..max_degree + 1)
        .map(|i| F::from_u128(i as u128))
        .collect();
    let weights = barycentric_weights(&points_vec);
    let mut expected_sum = sum;
    for round_index in 0..num_vars {
        let evals: Vec<F> = transcript
            .read_field_elements(max_degree + 1)
            .unwrap()
            .chunks(32)
            .map(|e| from_u8_to_f::<F>(e))
            .collect_vec();
        if evals.len() != max_degree + 1 {
            return false;
        }
        let round_poly_eval_at_0 = evals[0];
        let round_poly_eval_at_1 = evals[1];
        let computed_sum = round_poly_eval_at_0 + round_poly_eval_at_1;

        // Check r_{i}(α_i) == r_{i+1}(0) + r_{i+1}(1)
        if computed_sum != expected_sum {
            println!("computed_sum : {:?}", computed_sum);
            println!("expected_sum : {:?}", expected_sum);
            println!("round index : {}", round_index);
            return false;
        }

        let challenge = transcript.squeeze_challenge();

        // Compute r_{i}(α_i) using barycentric interpolation
        expected_sum = barycentric_interpolate(
            &weights,
            &points_vec,
            &evals,
            &challenge,
        );
    }
    true
}
