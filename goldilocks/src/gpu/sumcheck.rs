use std::cell::{RefCell, RefMut};

use cudarc::driver::{CudaView, CudaViewMut, DriverError, LaunchAsync, LaunchConfig};
use goldilocks::ExtensionField;

use crate::{FieldBinding, GPUSumcheckProver, QuadraticExtFieldBinding};

impl<E> GPUSumcheckProver<E>
where
    E: ExtensionField + From<QuadraticExtFieldBinding> + Into<QuadraticExtFieldBinding>,
    E::BaseField: From<FieldBinding> + Into<FieldBinding>,
{
    pub fn prove_sumcheck(
        &self,
        num_vars: usize,
        num_polys: usize,
        max_degree: usize,
        sum: E::BaseField,
        polys: &mut CudaViewMut<QuadraticExtFieldBinding>,
        device_ks: &[CudaView<FieldBinding>],
        buf: RefCell<CudaViewMut<QuadraticExtFieldBinding>>,
        challenges: &mut CudaViewMut<QuadraticExtFieldBinding>,
        round_evals: RefCell<CudaViewMut<QuadraticExtFieldBinding>>,
    ) -> Result<(), DriverError> {
        let initial_poly_num_vars = num_vars;
        for round in 0..num_vars {
            self.eval_at_k_and_combine(
                initial_poly_num_vars,
                round,
                max_degree,
                num_polys,
                &polys.slice(..),
                device_ks,
                buf.borrow_mut(),
                round_evals.borrow_mut(),
            )?;
            // squeeze challenge
            self.squeeze_challenge(round, challenges)?;
            // fold_into_half_in_place
            self.fold_into_half_in_place(
                initial_poly_num_vars,
                round,
                num_polys,
                polys,
                &challenges.slice(round..round + 1),
            )?;
        }
        Ok(())
    }

    pub(crate) fn eval_at_k_and_combine(
        &self,
        initial_poly_num_vars: usize,
        round: usize,
        max_degree: usize,
        num_polys: usize,
        polys: &CudaView<QuadraticExtFieldBinding>,
        device_ks: &[CudaView<FieldBinding>],
        mut buf: RefMut<CudaViewMut<QuadraticExtFieldBinding>>,
        mut round_evals: RefMut<CudaViewMut<QuadraticExtFieldBinding>>,
    ) -> Result<(), DriverError> {
        let (num_blocks, num_threads_per_block) = if initial_poly_num_vars - round <= 10 {
            (1, 1 << (initial_poly_num_vars - round))
        } else {
            (self.max_blocks_per_sm()? * self.num_sm()?, 1024)
        };
        for k in 0..max_degree + 1 {
            let eval_at_k_and_product = self
                .gpu
                .get_func("sumcheck", "eval_at_k_and_product")
                .unwrap();
            let launch_config = LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (num_threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                eval_at_k_and_product.launch(
                    launch_config,
                    (
                        initial_poly_num_vars - round,
                        1 << initial_poly_num_vars,
                        num_polys,
                        polys,
                        &mut *buf,
                        &device_ks[k],
                    ),
                )?;
            };
            let size = 1 << (initial_poly_num_vars - round - 1);
            let sum = self.gpu.get_func("sumcheck", "sum").unwrap();
            let launch_config = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (num_threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                sum.launch(
                    launch_config,
                    (
                        &mut *buf,
                        &mut *round_evals,
                        size >> 1,
                        round * (max_degree + 1) + k,
                    ),
                )?;
            };
        }
        Ok(())
    }

    pub(crate) fn squeeze_challenge(
        &self,
        round: usize,
        challenges: &mut CudaViewMut<QuadraticExtFieldBinding>,
    ) -> Result<(), DriverError> {
        let squeeze_challenge = self.gpu.get_func("sumcheck", "squeeze_challenge").unwrap();
        let launch_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            squeeze_challenge.launch(launch_config, (challenges, round))?;
        }
        Ok(())
    }

    pub(crate) fn fold_into_half_in_place(
        &self,
        initial_poly_num_vars: usize,
        round: usize,
        num_polys: usize,
        polys: &mut CudaViewMut<QuadraticExtFieldBinding>,
        challenge: &CudaView<QuadraticExtFieldBinding>,
    ) -> Result<(), DriverError> {
        let fold_into_half_in_place = self
            .gpu
            .get_func("sumcheck", "fold_into_half_in_place")
            .unwrap();
        let (num_blocks_per_poly, num_threads_per_block) = if initial_poly_num_vars - round <= 10 {
            (1, 1 << (initial_poly_num_vars - round))
        } else {
            (self.max_blocks_per_sm()? / num_polys * self.num_sm()?, 512)
        };
        let launch_config = LaunchConfig {
            grid_dim: ((num_blocks_per_poly * num_polys) as u32, 1, 1),
            block_dim: (num_threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            fold_into_half_in_place.launch(
                launch_config,
                (
                    initial_poly_num_vars - round,
                    1 << initial_poly_num_vars,
                    num_blocks_per_poly,
                    polys,
                    challenge,
                ),
            )?;
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, time::Instant};

    use cudarc::{
        driver::{
            result::event::{create, elapsed, record},
            sys, DriverError,
        },
        nvrtc::Ptx,
    };
    use ff::Field;
    use goldilocks::{ExtensionField, Goldilocks, GoldilocksExt2};
    use itertools::Itertools;
    use rand::rngs::OsRng;

    use crate::{cpu, GPUSumcheckProver, SUMCHECK_PTX};

    #[test]
    fn test_prove_sumcheck() -> Result<(), DriverError> {
        let nowall = Instant::now();
        let num_vars = 26;
        let num_polys = 3;
        let max_degree = 3;

        let rng = OsRng::default();
        let polys = (0..num_polys)
            .map(|_| {
                (0..1 << num_vars)
                    .map(|i| {
                        if i < 1024 {
                            GoldilocksExt2::from_base(&Goldilocks::random(rng))
                        } else {
                            GoldilocksExt2::from_base(&Goldilocks::from(i))
                        }
                    })
                    .collect_vec()
            })
            .collect_vec();

        let mut gpu_api_wrapper = GPUSumcheckProver::<GoldilocksExt2>::setup()?;
        gpu_api_wrapper.gpu.load_ptx(
            Ptx::from_src(SUMCHECK_PTX),
            "sumcheck",
            &[
                "eval_at_k_and_product",
                "fold_into_half_in_place",
                "sum",
                "squeeze_challenge",
            ],
        )?;

        let now = Instant::now();
        let mut gpu_polys = gpu_api_wrapper.copy_exts_to_device(&polys.concat())?;
        let sum = (0..1 << num_vars)
            .fold(GoldilocksExt2::ZERO, |acc, index| {
                acc + polys
                    .iter()
                    .map(|poly| poly[index])
                    .product::<GoldilocksExt2>()
            })
            .to_limbs()[0];
        let device_ks = (0..max_degree + 1)
            .map(|k| {
                gpu_api_wrapper
                    .gpu
                    .htod_copy(vec![Goldilocks::from(k as u64).into()])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut buf = gpu_api_wrapper.malloc_on_device(1 << (num_vars - 1))?;
        let buf_view = RefCell::new(buf.slice_mut(..));

        let mut challenges = gpu_api_wrapper.malloc_on_device(num_vars)?;
        let mut round_evals = gpu_api_wrapper.malloc_on_device(num_vars * (max_degree + 1))?;
        let round_evals_view = RefCell::new(round_evals.slice_mut(..));
        println!("Time taken to copy data to device : {:.2?}", now.elapsed());

        let start = create(sys::CUevent_flags::CU_EVENT_DEFAULT)?;
        let stop = create(sys::CUevent_flags::CU_EVENT_DEFAULT)?;
        let now = Instant::now();
        unsafe {
            record(start, *gpu_api_wrapper.gpu.cu_stream())?;
        }
        gpu_api_wrapper.prove_sumcheck(
            num_vars,
            num_polys,
            max_degree,
            sum,
            &mut gpu_polys.slice_mut(..),
            &device_ks
                .iter()
                .map(|device_k| device_k.slice(..))
                .collect_vec(),
            buf_view,
            &mut challenges.slice_mut(..),
            round_evals_view,
        )?;
        unsafe {
            record(stop, *gpu_api_wrapper.gpu.cu_stream())?;
        }
        gpu_api_wrapper.gpu.synchronize()?;
        let res = unsafe { elapsed(start, stop)? };
        println!("Time taken to prove sumcheck by GPU timer : {:.2?}", res);
        println!(
            "Time taken to prove sumcheck on gpu : {:.2?}",
            now.elapsed()
        );

        let challenges = gpu_api_wrapper.dtoh_sync_copy(&challenges.slice(..))?;
        let round_evals = (0..num_vars)
            .map(|i| {
                gpu_api_wrapper.dtoh_sync_copy(
                    &round_evals.slice(i * (max_degree + 1)..(i + 1) * (max_degree + 1)),
                )
            })
            .collect::<Result<Vec<Vec<GoldilocksExt2>>, _>>()?;
        let round_evals = round_evals
            .iter()
            .map(|round_evals| round_evals.as_slice())
            .collect_vec();
        let result = cpu::sumcheck::verify_sumcheck(
            num_vars,
            max_degree,
            sum,
            &challenges[..],
            &round_evals[..],
        );
        assert!(result);
        println!("Time taken to test : {:.2?}", nowall.elapsed());
        Ok(())
    }
}
