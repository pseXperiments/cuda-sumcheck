use std::cell::{RefCell, RefMut};

use cudarc::driver::{CudaView, CudaViewMut, DriverError, LaunchAsync, LaunchConfig};
use ff::PrimeField;

use crate::{FieldBinding, GPUApiWrapper};

impl<F: PrimeField + From<FieldBinding> + Into<FieldBinding>> GPUApiWrapper<F> {
    pub fn prove_sumcheck(
        &self,
        num_vars: usize,
        num_polys: usize,
        max_degree: usize,
        sum: F,
        polys: &mut CudaViewMut<FieldBinding>,
        device_ks: &[CudaView<FieldBinding>],
        buf: RefCell<CudaViewMut<FieldBinding>>,
        challenges: &mut CudaViewMut<FieldBinding>,
        round_evals: RefCell<CudaViewMut<FieldBinding>>,
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
        polys: &CudaView<FieldBinding>,
        device_ks: &[CudaView<FieldBinding>],
        mut buf: RefMut<CudaViewMut<FieldBinding>>,
        mut round_evals: RefMut<CudaViewMut<FieldBinding>>,
    ) -> Result<(), DriverError> {
        let (num_blocks_per_poly, num_threads_per_block) = if initial_poly_num_vars - round <= 10 {
            (1, 1 << (initial_poly_num_vars - round))
        } else {
            (self.max_blocks_per_sm()? / num_polys * self.num_sm()?, 512)
        };
        for k in 0..max_degree + 1 {
            let fold_into_half = self.gpu.get_func("sumcheck", "fold_into_half").unwrap();
            let launch_config = LaunchConfig {
                grid_dim: ((num_blocks_per_poly * num_polys) as u32, 1, 1),
                block_dim: (num_threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                fold_into_half.launch(
                    launch_config,
                    (
                        initial_poly_num_vars - round,
                        1 << initial_poly_num_vars,
                        num_blocks_per_poly,
                        polys,
                        &mut *buf,
                        &device_ks[k],
                    ),
                )?;
            };
            let size = 1 << (initial_poly_num_vars - round - 1);
            let combine = self.gpu.get_func("sumcheck", "combine").unwrap();
            let launch_config = LaunchConfig {
                grid_dim: (num_blocks_per_poly as u32, 1, 1),
                block_dim: (num_threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                combine.launch(launch_config, (&mut *buf, size, num_polys))?;
            };
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
        challenges: &mut CudaViewMut<FieldBinding>,
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
        polys: &mut CudaViewMut<FieldBinding>,
        challenge: &CudaView<FieldBinding>,
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
            result::{
                event::{create, elapsed, record},
                stream::wait_event,
            },
            sys, DriverError,
        },
        nvrtc::Ptx,
    };
    use ff::Field;
    use goldilocks::Goldilocks;
    use itertools::Itertools;
    use rand::rngs::OsRng;

    use crate::{cpu, GPUApiWrapper, SUMCHECK_PTX};

    #[test]
    fn test_eval_at_k_and_combine() -> Result<(), DriverError> {
        let num_vars = 10;
        let num_polys = 3;
        let max_degree = 3;
        let rng = OsRng::default();

        let combine_function = |args: &Vec<Goldilocks>| args.iter().product();

        let polys = (0..num_polys)
            .map(|_| {
                (0..1 << num_vars)
                    .map(|i| {
                        if i < 1024 {
                            Goldilocks::random(rng)
                        } else {
                            Goldilocks::from(i)
                        }
                    })
                    .collect_vec()
            })
            .collect_vec();

        let mut gpu_api_wrapper = GPUApiWrapper::<Goldilocks>::setup()?;
        gpu_api_wrapper.gpu.load_ptx(
            Ptx::from_src(SUMCHECK_PTX),
            "sumcheck",
            &["fold_into_half", "combine", "sum"],
        )?;

        let mut cpu_round_evals = vec![];
        let now = Instant::now();
        let polys = polys.iter().map(|poly| poly.as_slice()).collect_vec();
        for k in 0..max_degree + 1 {
            cpu_round_evals.push(cpu::sumcheck::eval_at_k_and_combine(
                num_vars,
                polys.as_slice(),
                &combine_function,
                Goldilocks::from(k),
            ));
        }
        println!(
            "Time taken to eval_at_k_and_combine on cpu: {:.2?}",
            now.elapsed()
        );

        // copy polynomials to device
        let gpu_polys = gpu_api_wrapper.copy_to_device(&polys.concat())?;
        let device_ks = (0..max_degree + 1)
            .map(|k| {
                gpu_api_wrapper
                    .gpu
                    .htod_copy(vec![Goldilocks::from(k as u64).into()])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut buf = gpu_api_wrapper.malloc_on_device(num_polys << (num_vars - 1))?;
        let buf_view = RefCell::new(buf.slice_mut(..));
        let mut round_evals = gpu_api_wrapper.malloc_on_device(max_degree as usize + 1)?;
        let round_evals_view = RefCell::new(round_evals.slice_mut(..));
        let round = 0;
        let now = Instant::now();
        gpu_api_wrapper.eval_at_k_and_combine(
            num_vars,
            round,
            max_degree as usize,
            num_polys,
            &gpu_polys.slice(..),
            &device_ks
                .iter()
                .map(|device_k| device_k.slice(..))
                .collect_vec(),
            buf_view.borrow_mut(),
            round_evals_view.borrow_mut(),
        )?;
        println!(
            "Time taken to eval_at_k_and_combine on gpu: {:.2?}",
            now.elapsed()
        );
        let gpu_round_evals =
            gpu_api_wrapper.dtoh_sync_copy(&round_evals.slice(0..(max_degree + 1) as usize))?;
        cpu_round_evals
            .iter()
            .zip_eq(gpu_round_evals.iter())
            .for_each(|(a, b)| {
                assert_eq!(a, b);
            });

        Ok(())
    }

    #[test]
    fn test_fold_into_half_in_place() -> Result<(), DriverError> {
        let num_vars = 6;
        let num_polys = 4;

        let rng = OsRng::default();
        let mut polys = (0..num_polys)
            .map(|_| {
                (0..1 << num_vars)
                    .map(|i| {
                        if i < 1024 {
                            Goldilocks::random(rng)
                        } else {
                            Goldilocks::from(i)
                        }
                    })
                    .collect_vec()
            })
            .collect_vec();

        let mut gpu_api_wrapper = GPUApiWrapper::<Goldilocks>::setup()?;
        gpu_api_wrapper.gpu.load_ptx(
            Ptx::from_src(SUMCHECK_PTX),
            "sumcheck",
            &["fold_into_half_in_place"],
        )?;
        // copy polynomials to device
        let mut gpu_polys = gpu_api_wrapper.copy_to_device(&polys.concat())?;
        let challenge = Goldilocks::random(rng);
        let gpu_challenge = gpu_api_wrapper.copy_to_device(&vec![challenge])?;
        let round = 0;

        let now = Instant::now();
        gpu_api_wrapper.fold_into_half_in_place(
            num_vars,
            round,
            num_polys,
            &mut gpu_polys.slice_mut(..),
            &gpu_challenge.slice(..),
        )?;
        println!(
            "Time taken to fold_into_half_in_place on gpu: {:.2?}",
            now.elapsed()
        );

        let gpu_result = (0..num_polys)
            .map(|i| {
                gpu_api_wrapper
                    .dtoh_sync_copy(&gpu_polys.slice(i << num_vars..(i * 2 + 1) << (num_vars - 1)))
            })
            .collect::<Result<Vec<Vec<Goldilocks>>, _>>()?;

        let now = Instant::now();
        (0..num_polys)
            .for_each(|i| cpu::sumcheck::fold_into_half_in_place(&mut polys[i], challenge));
        println!("Time taken to fold_into_half on cpu: {:.2?}", now.elapsed());
        polys
            .iter_mut()
            .for_each(|poly| poly.truncate(1 << (num_vars - 1)));

        gpu_result
            .into_iter()
            .zip_eq(polys)
            .for_each(|(gpu_result, cpu_result)| {
                assert_eq!(gpu_result, cpu_result);
            });

        Ok(())
    }

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
                            Goldilocks::random(rng)
                        } else {
                            Goldilocks::from(i)
                        }
                    })
                    .collect_vec()
            })
            .collect_vec();

        let mut gpu_api_wrapper = GPUApiWrapper::<Goldilocks>::setup()?;
        gpu_api_wrapper.gpu.load_ptx(
            Ptx::from_src(SUMCHECK_PTX),
            "sumcheck",
            &[
                "fold_into_half",
                "fold_into_half_in_place",
                "combine",
                "sum",
                "squeeze_challenge",
            ],
        )?;

        let now = Instant::now();
        let mut gpu_polys = gpu_api_wrapper.copy_to_device(&polys.concat())?;
        let sum = (0..1 << num_vars).fold(Goldilocks::ZERO, |acc, index| {
            acc + polys.iter().map(|poly| poly[index]).product::<Goldilocks>()
        });
        let device_ks = (0..max_degree + 1)
            .map(|k| {
                gpu_api_wrapper
                    .gpu
                    .htod_copy(vec![Goldilocks::from(k as u64).into()])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut buf = gpu_api_wrapper.malloc_on_device(num_polys << (num_vars - 1))?;
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
            .collect::<Result<Vec<Vec<Goldilocks>>, _>>()?;
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
