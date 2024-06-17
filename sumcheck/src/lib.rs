use std::marker::PhantomData;
use std::time::Instant;
use cudarc::driver::{CudaDevice, LaunchConfig, DeviceRepr, DriverError, LaunchAsync};
use cudarc::nvrtc::Ptx;
use ff::{Field, PrimeField};
use halo2curves::bn256::Fr;
use itertools::Itertools;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// include the compiled PTX code as string
const CUDA_KERNEL_MY_STRUCT: &str = include_str!(concat!(env!("OUT_DIR"), "/multilinear.ptx"));

unsafe impl DeviceRepr for FieldBinding {}
impl Default for FieldBinding {
    fn default() -> Self {
        Self{ data: [0; 4] }
    }
}

impl<F: PrimeField> From<F> for FieldBinding {
    fn from(value: F) -> Self {
        let repr = value.to_repr();
        let bytes = repr.as_ref();
        let data = bytes.chunks(8).map(|bytes| {
            u64::from_le_bytes(bytes.try_into().unwrap())
        }).collect_vec();
        FieldBinding {
            data: data.try_into().unwrap()
        }
    }
}

/// Wrapper struct for APIs using GPU
#[derive(Default)]
pub struct GPUApiWrapper<F: Field>(PhantomData<F>);

impl<F: Field> GPUApiWrapper<F> {
    pub fn evaluate_poly(&self, poly_coeffs: &[F], point: &[F]) -> Result<F, DriverError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::{default, time::Instant};

    use cudarc::{driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig}, nvrtc::Ptx};
    use ff::{Field, PrimeField};
    use halo2curves::bn256::Fr;
    use itertools::Itertools;
    use rand::rngs::OsRng;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

    use super::{GPUApiWrapper, FieldBinding, CUDA_KERNEL_MY_STRUCT};

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

        let a_data = FieldBinding {
            data: [2, 0, 0, 0]
        };

        let b_data = FieldBinding {
            data: [
                0xa1f0fac9f8000001,
                0x9419f4243cdcb848,
                0xdc2822db40c0ac2e,
                0x183227397098d014,
            ]
        };

        // copy to GPU
        let gpu_field_structs = gpu.htod_copy(vec![a_data, b_data])?;
        let results = gpu.htod_copy(vec![FieldBinding::default(); 1024])?;

        println!("Time taken to initialise data: {:.2?}", now.elapsed());

        let now = Instant::now();

        let f = gpu.get_func("my_module", "mul").unwrap();

        unsafe { f.launch(LaunchConfig::for_num_elems(1024 as u32), (&gpu_field_structs, &results)) }?;

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
