use cudarc::driver::{CudaDevice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use ff::PrimeField;
use field::{FromFieldBinding, ToFieldBinding};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::marker::PhantomData;
use std::sync::Arc;
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
const SCALAR_MULTIPLICATION_KERNEL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/scalar_multiplication.ptx"));

/// Wrapper struct for APIs using GPU
pub struct GPUApiWrapper<F: PrimeField + FromFieldBinding<F> + ToFieldBinding<F>> {
    gpu: Arc<CudaDevice>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + FromFieldBinding<F> + ToFieldBinding<F>> GPUApiWrapper<F> {
    pub fn setup() -> Result<Self, DriverError> {
        // setup GPU device
        let now = Instant::now();
        let gpu = CudaDevice::new(0)?;
        println!("Time taken to initialise CUDA: {:.2?}", now.elapsed());
        Ok(Self {
            gpu,
            _marker: PhantomData,
        })
    }

    pub fn load_ptx(
        &self,
        ptx: &str,
        module_name: &str,
        func_names: &[&'static str],
    ) -> Result<(), DriverError> {
        // compile ptx
        let now = Instant::now();
        let ptx = Ptx::from_src(ptx);
        self.gpu.load_ptx(ptx, module_name, &func_names)?;
        println!("Time taken to compile and load PTX: {:.2?}", now.elapsed());
        Ok(())
    }

    pub fn evaluate_poly(
        &self,
        num_vars: usize,
        poly_coeffs: &[F],
        point: &[F],
    ) -> Result<F, DriverError> {
        let now = Instant::now();
        let point_montgomery = point
            .into_iter()
            .map(|f| F::to_montgomery_form(*f))
            .collect_vec();

        // copy to GPU
        let gpu_coeffs = self.gpu.htod_copy(
            poly_coeffs
                .into_par_iter()
                .map(|&coeff| F::to_canonical_form(coeff))
                .collect(),
        )?;
        let gpu_eval_point = self.gpu.htod_copy(point_montgomery)?;
        let monomial_evals = self
            .gpu
            .htod_copy(vec![FieldBinding::default(); 1 << num_vars])?;
        println!("Time taken to initialise data: {:.2?}", now.elapsed());

        let now = Instant::now();
        let evaluate_optimized = self
            .gpu
            .get_func("multilinear", "evaluate_optimized")
            .unwrap();

        unsafe {
            evaluate_optimized.launch(
                LaunchConfig::for_num_elems(1 << num_vars as u32),
                (&gpu_coeffs, &gpu_eval_point, num_vars, &monomial_evals),
            )?;
        };
        println!("Time taken to call kernel: {:.2?}", now.elapsed());

        let now = Instant::now();
        let monomial_evals = self.gpu.sync_reclaim(monomial_evals)?;
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

    pub fn mul(&self, values: &[F; 2]) -> Result<F, DriverError> {
        let now = Instant::now();
        let gpu_values = self
            .gpu
            .htod_copy(values.map(|v| F::to_montgomery_form(v)).to_vec())?;
        let results = self.gpu.htod_copy(vec![FieldBinding::default(); 1])?;
        println!("Time taken to initialise data: {:.2?}", now.elapsed());

        let now = Instant::now();
        let mul = self.gpu.get_func("scalar_multiplication", "mul").unwrap();
        unsafe {
            mul.launch(
                LaunchConfig::for_num_elems(1 as u32),
                (&gpu_values, &results)
            )?;
        }
        println!("Time taken to call kernel: {:.2?}", now.elapsed());

        let now = Instant::now();
        let results = self.gpu.sync_reclaim(results)?;
        println!("Time taken to synchronize: {:.2?}", now.elapsed());
        Ok(F::from_canonical_form(results[0]))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use cudarc::driver::DriverError;
    use ff::{Field, PrimeField};
    use halo2curves::bn256::Fr;
    use itertools::Itertools;
    use rand::rngs::OsRng;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

    use crate::{MULTILINEAR_POLY_KERNEL, SCALAR_MULTIPLICATION_KERNEL};

    use super::GPUApiWrapper;

    fn evaluate_poly_cpu<F: PrimeField>(poly_coeffs: &[F], point: &[F], num_vars: usize) -> F {
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
        let num_vars = 18;
        let rng = OsRng::default();
        let poly_coeffs = (0..1 << num_vars).map(|_| Fr::random(rng)).collect_vec();
        let point = (0..num_vars).map(|_| Fr::random(rng)).collect_vec();

        let gpu_api_wrapper = GPUApiWrapper::<Fr>::setup()?;
        gpu_api_wrapper.load_ptx(
            MULTILINEAR_POLY_KERNEL,
            "multilinear",
            &["evaluate_optimized"],
        )?;

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
        let rng = OsRng::default();
        let values = [(); 2].map(|_| Fr::random(rng));
        let expected = values[0] * values[1];

        let gpu_api_wrapper = GPUApiWrapper::<Fr>::setup()?;
        gpu_api_wrapper.load_ptx(
            SCALAR_MULTIPLICATION_KERNEL,
            "scalar_multiplication",
            &["mul"],
        )?;

        let now = Instant::now();
        let actual = gpu_api_wrapper.mul(&values)?;
        println!("Time taken to evaluate on gpu: {:.2?}", now.elapsed());

        assert_eq!(actual, expected);
        Ok(())
    }
}
