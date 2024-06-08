use std::time::Instant;
use cudarc::driver::{CudaDevice, LaunchConfig, DeviceRepr, DriverError, LaunchAsync};
use cudarc::nvrtc::Ptx;
use ff::Field;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// TODO : Replace this with our sumcheck struct
unsafe impl DeviceRepr for MyStruct {}
impl Default for MyStruct {
    fn default() -> Self{
        Self{ data: [0.0; 4]}
    }
}

// include the compiled PTX code as string
const CUDA_KERNEL_MY_STRUCT: &str = include_str!(concat!(env!("OUT_DIR"), "/sumcheck.ptx"));

/// Wrapper struct for APIs using GPU
#[derive(Default)]
pub struct GPUApiWrapper<F: Field> {}

impl<F: Field> GPUApiWrapper<F> {
    pub fn evaluate_poly(&self, poly_coeffs: &[F], point: &[F]) -> Result<F, DriverError> {

    }
}

#[cfg(test)]
mod tests {
    use std::default;

    use cudarc::driver::DriverError;
    use ff::Field;
    use halo2curves::bn256::Fr;
    use itertools::Itertools;
    use rand::rngs::OsRng;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

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
                    let mut result = F::ONE;
                    for (index, point) in indices.iter().zip(point.iter()) {
                        result *= if *index == 1 { *point } else { F::ONE };
                    }
                    result * coeff
                }
            })
            .sum()
    }

    #[test]
    fn test_evaluate_poly() -> Result<(), DriverError> {
        let num_vars = 16;
        let rng = OsRng::default();
        let poly_coeffs = (0..1 << num_vars).map(|_| {
            Fr::random(rng)
        }).collect_vec();
        let point = (0..num_vars).map(|_| {
            Fr::random(rng)
        }).collect_vec();
        let gpu_api_wrapper = GPUApiWrapper::<Fr>::default();
        let eval_poly_result_by_cpu = evaluate_poly_cpu(&poly_coeffs, &point, num_vars);
        let eval_poly_result_by_gpu = gpu_api_wrapper.evaluate_poly(&poly_coeffs, &point)?;
        assert_eq!(eval_poly_result_by_cpu, eval_poly_result_by_gpu);
        Ok(())
    }
}
