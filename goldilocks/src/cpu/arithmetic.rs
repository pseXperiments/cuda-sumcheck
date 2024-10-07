use ff::{BatchInvert, Field};
use itertools::Itertools;
use num_integer::Integer;

pub fn usize_from_bits_le(bits: &[bool]) -> usize {
    bits.iter()
        .rev()
        .fold(0, |int, bit| (int << 1) + (*bit as usize))
}

pub fn div_ceil(dividend: usize, divisor: usize) -> usize {
    Integer::div_ceil(&dividend, &divisor)
}

pub fn barycentric_weights<F: Field>(points: &[F]) -> Vec<F> {
    let mut weights = points
        .iter()
        .enumerate()
        .map(|(j, point_j)| {
            points
                .iter()
                .enumerate()
                .filter(|(i, _)| i != &j)
                .map(|(_, point_i)| *point_j - point_i)
                .reduce(|acc, value| acc * &value)
                .unwrap_or(F::ONE)
        })
        .collect_vec();
    weights.batch_invert();
    weights
}

pub fn inner_product<'a, 'b, F: Field>(
    lhs: impl IntoIterator<Item = &'a F>,
    rhs: impl IntoIterator<Item = &'b F>,
) -> F {
    lhs.into_iter()
        .zip_eq(rhs)
        .map(|(lhs, rhs)| *lhs * rhs)
        .reduce(|acc, product| acc + product)
        .unwrap_or_default()
}

pub fn barycentric_interpolate<F: Field>(weights: &[F], points: &[F], evals: &[F], x: &F) -> F {
    let (coeffs, sum_inv) = {
        let mut coeffs = points.iter().map(|point| *x - point).collect_vec();
        coeffs.batch_invert();
        coeffs.iter_mut().zip(weights).for_each(|(coeff, weight)| {
            *coeff *= weight;
        });
        let sum_inv = coeffs.iter().fold(F::ZERO, |sum, coeff| sum + coeff);
        (coeffs, sum_inv.invert().unwrap())
    };
    inner_product(&coeffs, evals) * &sum_inv
}
