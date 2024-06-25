use cudarc::driver::{CudaDevice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use ff::PrimeField;
use field::{FromFieldBinding, ToFieldBinding};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::marker::PhantomData;
use std::time::Instant;

pub mod field;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe impl DeviceRepr for FieldBinding {}
impl Default for FieldBinding {
    fn default() -> Self {
        Self { data: [0; 4] }
    }
}

// include the compiled PTX code as string
const MULTILINEAR_POLY_KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/multilinear.ptx"));

/// Wrapper struct for APIs using GPU
#[derive(Default)]
pub struct GPUApiWrapper<F: PrimeField + FromFieldBinding<F> + ToFieldBinding<F>>(PhantomData<F>);

impl<F: PrimeField + FromFieldBinding<F> + ToFieldBinding<F>> GPUApiWrapper<F> {
    pub fn evaluate_poly(
        &self,
        num_vars: usize,
        poly_coeffs: &[F],
        point: &[F],
    ) -> Result<F, DriverError> {
        // setup GPU device
        let now = Instant::now();

        let gpu = CudaDevice::new(0)?;

        println!("Time taken to initialise CUDA: {:.2?}", now.elapsed());

        // compile ptx
        let now = Instant::now();

        let ptx = Ptx::from_src(MULTILINEAR_POLY_KERNEL);
        gpu.load_ptx(ptx, "multilinear", &["evaluate", "evaluate_optimized"])?;

        println!("Time taken to compile and load PTX: {:.2?}", now.elapsed());

        let now = Instant::now();
        let point_montgomery = point
            .into_iter()
            .map(|f| F::to_montgomery_form(*f))
            .collect_vec();

        // copy to GPU
        let gpu_coeffs = gpu.htod_copy(
            poly_coeffs
                .into_par_iter()
                .map(|&coeff| F::to_canonical_form(coeff))
                .collect(),
        )?;
        let gpu_eval_point = gpu.htod_copy(point_montgomery)?;
        let monomial_evals = gpu.htod_copy(vec![FieldBinding::default(); 1 << num_vars])?;
        let mutex = gpu.alloc_zeros::<u32>(1)?;
        let result = gpu.htod_copy(vec![FieldBinding::default(); 1])?;

        println!("Time taken to initialise data: {:.2?}", now.elapsed());

        let now = Instant::now();

        let evaluate_optimized = gpu.get_func("multilinear", "evaluate_optimized").unwrap();

        unsafe {
            evaluate_optimized.launch(
                LaunchConfig::for_num_elems(1 << num_vars as u32),
                (
                    &gpu_coeffs,
                    &gpu_eval_point,
                    num_vars,
                    &monomial_evals,
                    &result,
                    &mutex,
                ),
            )?;
        };

        println!("Time taken to call kernel: {:.2?}", now.elapsed());

        let now = Instant::now();
        let monomial_evals = gpu.sync_reclaim(monomial_evals)?;
        println!("Time taken to synchronize: {:.2?}", now.elapsed());

        let now = Instant::now();
        let result = monomial_evals
            .into_iter()
            .step_by(1024)
            .map(|eval| F::from_canonical_form(eval))
            .sum::<F>();
        println!("Time taken to calculate sum: {:.2?}", now.elapsed());
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp::Ordering, default, fmt::Error, time::Instant};

    use cudarc::{
        driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
        nvrtc::Ptx,
    };
    use ff::{Field, PrimeField};
    use halo2curves::bn256::Fr;
    use itertools::Itertools;
    use rand::rngs::OsRng;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

    use crate::{field::{FromFieldBinding, ToFieldBinding}, FieldBinding, MULTILINEAR_POLY_KERNEL};

    use super::GPUApiWrapper;

    fn evaluate_poly_cpu<F: Field>(poly_coeffs: &[F], point: &[F], num_vars: usize) -> F {
        poly_coeffs
            .par_iter()
            .enumerate()
            .map(|(i, coeff)| {
                if *coeff == F::ZERO {
                    F::ZERO
                } else {
                    let indices = (0..num_vars).map(|j| (i >> j) & 1).collect_vec();
                    let mut result = coeff.clone();
                    for (index, point) in indices.iter().zip(point.iter()) {
                        result *= if *index == 1 { *point } else { F::ONE };
                    }
                    result
                }
            })
            .sum()
    }

    #[test]
    fn test_evaluate_poly() -> Result<(), DriverError> {
        let num_vars = 22;
        let rng = OsRng::default();
        let poly_coeffs = (0..1 << num_vars).map(|_| Fr::random(rng)).collect_vec();
        let point = (0..num_vars).map(|_| Fr::random(rng)).collect_vec();
        let gpu_api_wrapper = GPUApiWrapper::<Fr>::default();
        let now = Instant::now();
        let eval_poly_result_by_cpu = evaluate_poly_cpu(&poly_coeffs, &point, num_vars);
        println!("Time taken to evaluate on cpu: {:.2?}", now.elapsed());

        let now = Instant::now();
        let eval_poly_result_by_gpu =
            gpu_api_wrapper.evaluate_poly(num_vars, &poly_coeffs, &point)?;
        println!("Time taken to evaluate on gpu: {:.2?}", now.elapsed());

        assert_eq!(eval_poly_result_by_cpu, eval_poly_result_by_gpu);
        Ok(())
    }

    #[test]
    fn test_scalar_multiplication() -> Result<(), DriverError> {
        // setup GPU device
        let now = Instant::now();

        let gpu = CudaDevice::new(0)?;

        println!("Time taken to initialise CUDA: {:.2?}", now.elapsed());

        // compile ptx
        let now = Instant::now();

        let ptx = Ptx::from_src(CUDA_KERNEL_MY_STRUCT);
        gpu.load_ptx(ptx, "my_module", &["mul"])?;

        println!("Time taken to compile and load PTX: {:.2?}", now.elapsed());

        let a = Fr::from(2);
        let b = Fr::TWO_INV;

        println!("a * b : {:?}", a * b);

        let a_data = FieldBinding { data: [2, 0, 0, 0] };

        let b_data = FieldBinding {
            data: [
                0xa1f0fac9f8000001,
                0x9419f4243cdcb848,
                0xdc2822db40c0ac2e,
                0x183227397098d014,
            ],
        };

        // copy to GPU
        let gpu_field_structs = gpu.htod_copy(vec![a_data, b_data])?;
        let results = gpu.htod_copy(vec![FieldBinding::default(); 1024])?;

        println!("Time taken to initialise data: {:.2?}", now.elapsed());

        let now = Instant::now();

        let f = gpu.get_func("my_module", "mul").unwrap();

        unsafe {
            f.launch(
                LaunchConfig::for_num_elems(1024 as u32),
                (&gpu_field_structs, &results),
            )
        }?;

        println!("Time taken to call kernel: {:.2?}", now.elapsed());

        let results = gpu.sync_reclaim(results)?;

        results.iter().for_each(|result| {
            assert_eq!(result.data[0], 1);
            assert_eq!(result.data[1], 0);
            assert_eq!(result.data[1], 0);
            assert_eq!(result.data[1], 0);
        });
        Ok(())
    }
}
