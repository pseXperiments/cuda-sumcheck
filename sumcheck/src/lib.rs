// silence warnings due to bindgen
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use ff::PrimeField;
use field::{FromFieldBinding, ToFieldBinding};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

mod cpu;
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

    pub fn convert_to_montgomery(
        &self,
        values: &[F],
        size: usize,
        chunk_size: usize,
    ) -> Result<CudaSlice<FieldBinding>, DriverError> {
        let now = Instant::now();
        let values = self.gpu.htod_copy(
            values
                .into_par_iter()
                .map(|&eval| F::to_canonical_form(eval))
                .collect(),
        )?;
        println!("Time taken to initialise data: {:.2?}", now.elapsed());
        let now = Instant::now();
        let convert_to_montgomery = self
            .gpu
            .get_func("multilinear", "convert_to_montgomery")
            .unwrap();
        unsafe {
            convert_to_montgomery.launch(
                LaunchConfig::for_num_elems((size / chunk_size) as u32),
                (&values, size, chunk_size),
            )?;
        };
        println!("Time taken to call kernel: {:.2?}", now.elapsed());
        self.gpu.synchronize()?;
        Ok(values)
    }

    pub fn eval(&self, num_vars: usize, evals: &[F], point: &[F]) -> Result<F, DriverError> {
        let now = Instant::now();
        let point = point
            .into_iter()
            .map(|f| F::to_montgomery_form(*f))
            .collect_vec();

        // copy to GPU
        let gpu_eval_point = self.gpu.htod_copy(point)?;
        let mut evals = self.convert_to_montgomery(evals, 1 << num_vars, 1 << 5)?;

        let mut num_vars = num_vars;
        let mut results = vec![];
        let mut offset = 0;
        while num_vars > 0 {
            let log2_chunk_size = 2;
            let chunk_size = 1 << log2_chunk_size;
            let (data_size_per_block, block_num) = if num_vars < 10 + log2_chunk_size {
                (1 << num_vars, 1)
            } else {
                (
                    1 << (10 + log2_chunk_size),
                    1 << (num_vars - 10 - log2_chunk_size),
                )
            };
            // each block produces single result and store to `buf`
            let buf = self.gpu.htod_copy(vec![
                FieldBinding::default();
                1 << (num_vars - log2_chunk_size)
            ])?;
            println!("Time taken to initialise data: {:.2?}", now.elapsed());
            let now = Instant::now();
            let eval = self.gpu.get_func("multilinear", "eval").unwrap();
            unsafe {
                eval.launch(
                    LaunchConfig::for_num_elems((block_num << 10) as u32),
                    (
                        &evals,
                        &buf,
                        &gpu_eval_point,
                        data_size_per_block,
                        chunk_size,
                        offset,
                    ),
                )?;
            };
            println!("Time taken to call kernel: {:.2?}", now.elapsed());
            let now = Instant::now();
            results = self.gpu.sync_reclaim(buf)?;
            println!("Time taken to synchronize: {:.2?}", now.elapsed());
            if block_num == 1 {
                break;
            } else {
                evals = self
                    .gpu
                    .htod_copy(results.iter().cloned().step_by(1 << 10).collect_vec())?;
                num_vars -= 10 + log2_chunk_size;
                offset += 10 + log2_chunk_size;
            }
        }

        Ok(F::from_montgomery_form(results[0]))
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
                (&gpu_values, &results),
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
    use ff::Field;
    use halo2curves::bn256::Fr;
    use itertools::Itertools;
    use rand::rngs::OsRng;

    use crate::{cpu, MULTILINEAR_POLY_KERNEL, SCALAR_MULTIPLICATION_KERNEL};

    use super::GPUApiWrapper;

    #[test]
    fn test_eval() -> Result<(), DriverError> {
        let num_vars = 22;
        let rng = OsRng::default();
        let evals = (0..1 << num_vars).map(|_| Fr::random(rng)).collect_vec();
        let point = (0..num_vars).map(|_| Fr::random(rng)).collect_vec();

        let gpu_api_wrapper = GPUApiWrapper::<Fr>::setup()?;
        gpu_api_wrapper.load_ptx(
            MULTILINEAR_POLY_KERNEL,
            "multilinear",
            &["convert_to_montgomery", "eval"],
        )?;

        let now = Instant::now();
        let eval_poly_result_by_cpu = cpu::multilinear::evaluate(&evals, &point);
        println!("Time taken to evaluate on cpu: {:.2?}", now.elapsed());

        let now = Instant::now();
        let eval_poly_result_by_gpu = gpu_api_wrapper.eval(num_vars, &evals, &point)?;
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
