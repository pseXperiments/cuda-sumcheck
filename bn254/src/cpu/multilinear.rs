use std::{borrow::Cow, mem};

use ff::Field;

use crate::cpu::{arithmetic::usize_from_bits_le, parallel::parallelize};

macro_rules! zip_self {
    (@ $iter:expr, $step:expr, $skip:expr) => {
        $iter.skip($skip).step_by($step).zip($iter.skip($skip + ($step >> 1)).step_by($step))
    };
    ($iter:expr) => {
        zip_self!(@ $iter, 2, 0)
    };
    ($iter:expr, $step:expr) => {
        zip_self!(@ $iter, $step, 0)
    };
    ($iter:expr, $step:expr, $skip:expr) => {
        zip_self!(@ $iter, $step, $skip)
    };
}

fn merge_into<F: Field>(target: &mut Vec<F>, evals: &[F], x_i: &F, distance: usize, skip: usize) {
    assert!(target.capacity() >= evals.len() >> distance);
    target.resize(evals.len() >> distance, F::ZERO);

    let step = 1 << distance;
    parallelize(target, |(target, start)| {
        let start = (start << distance) + skip;
        for (target, (eval_0, eval_1)) in
            target.iter_mut().zip(zip_self!(evals.iter(), step, start))
        {
            *target = (*eval_1 - eval_0) * x_i + eval_0;
        }
    });
}

fn merge_in_place<F: Field>(
    evals: &mut Cow<[F]>,
    x_i: &F,
    distance: usize,
    skip: usize,
    buf: &mut Vec<F>,
) {
    merge_into(buf, evals, x_i, distance, skip);
    if let Cow::Owned(_) = evals {
        mem::swap(evals.to_mut(), buf);
    } else {
        *evals = mem::replace(buf, Vec::with_capacity(buf.len() >> 1)).into();
    }
}

pub fn evaluate<F: Field>(evals: &[F], x: &[F]) -> F {
    assert_eq!(1 << x.len(), evals.len());

    let mut evals = Cow::Borrowed(evals);
    let mut bits = Vec::new();
    let mut buf = Vec::with_capacity(evals.len() >> 1);
    for x_i in x.iter() {
        if x_i == &F::ZERO || x_i == &F::ONE {
            bits.push(x_i == &F::ONE);
            continue;
        }

        let distance = bits.len() + 1;
        let skip = usize_from_bits_le(&bits);
        merge_in_place(&mut evals, x_i, distance, skip, &mut buf);
        bits.clear();
    }

    evals[usize_from_bits_le(&bits)]
}
