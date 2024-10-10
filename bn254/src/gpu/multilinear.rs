use std::time::Instant;

use cudarc::{
    driver::{DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};
use ff::PrimeField;

use crate::{fieldbinding::FieldBindingConversion, GPUApiWrapper, MULTILINEAR_PTX};

impl<F: PrimeField + FieldBindingConversion<F>> GPUApiWrapper<F> {
    pub fn eval(&mut self, num_vars: usize, evals: &[F], point: &[F]) -> Result<F, DriverError> {
        self.gpu.load_ptx(
            Ptx::from_src(MULTILINEAR_PTX),
            "multilinear",
            &["convert_to_montgomery_form", "eval"],
        )?;
        let now = Instant::now();
        // copy to GPU
        let gpu_eval_point = self.copy_to_device(point)?;
        let evals = self.copy_to_device(evals)?;
        println!("Time taken to initialise data: {:.2?}", now.elapsed());

        let mut num_vars = num_vars;
        let mut results = vec![];
        let mut offset = 0;
        while num_vars > 0 {
            let log2_data_size_per_block = 10;
            let (data_size_per_block, num_blocks) = if num_vars < log2_data_size_per_block {
                (1 << num_vars, 1)
            } else {
                (
                    1 << log2_data_size_per_block,
                    1 << (num_vars - log2_data_size_per_block),
                )
            };
            let now = Instant::now();
            let eval = self.gpu.get_func("multilinear", "eval").unwrap();
            // (number of field elements processed per thread block) * 32
            let shared_mem_bytes = data_size_per_block << 5;
            let launch_config = LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes,
            };
            unsafe {
                eval.launch(
                    launch_config,
                    (
                        &evals,
                        &gpu_eval_point,
                        data_size_per_block,
                        offset,
                        num_blocks,
                    ),
                )?;
            };
            println!("Time taken to call kernel: {:.2?}", now.elapsed());
            if num_blocks == 1 {
                let now = Instant::now();
                results = self.gpu.sync_reclaim(evals)?;
                println!("Time taken to synchronize: {:.2?}", now.elapsed());
                break;
            } else {
                num_vars -= log2_data_size_per_block;
                offset += log2_data_size_per_block;
            }
        }

        Ok(F::from_montgomery_form(results[0]))
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

    use crate::{cpu, GPUApiWrapper};

    #[test]
    fn test_eval() -> Result<(), DriverError> {
        let num_vars = 25;
        let rng = OsRng::default();
        let evals = (0..1 << num_vars).map(|_| Fr::random(rng)).collect_vec();
        let point = (0..num_vars).map(|_| Fr::random(rng)).collect_vec();

        let mut gpu_api_wrapper = GPUApiWrapper::<Fr>::setup()?;

        let now = Instant::now();
        let eval_poly_result_by_cpu = cpu::multilinear::evaluate(&evals, &point);
        println!("Time taken to evaluate on cpu: {:.2?}", now.elapsed());

        let now = Instant::now();
        let eval_poly_result_by_gpu = gpu_api_wrapper.eval(num_vars, &evals, &point)?;
        println!("Time taken to evaluate on gpu: {:.2?}", now.elapsed());

        assert_eq!(eval_poly_result_by_cpu, eval_poly_result_by_gpu);
        Ok(())
    }
}
