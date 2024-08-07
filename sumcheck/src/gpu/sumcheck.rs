use std::cell::{RefCell, RefMut};

use cudarc::driver::{CudaView, CudaViewMut, DriverError, LaunchAsync, LaunchConfig};
use ff::PrimeField;

use crate::{
    fieldbinding::{FromFieldBinding, ToFieldBinding},
    FieldBinding, GPUApiWrapper,
};

impl<F: PrimeField + FromFieldBinding<F> + ToFieldBinding<F>> GPUApiWrapper<F> {
    pub fn prove_sumcheck(
        &self,
        num_vars: usize,
        num_polys: usize,
        max_degree: usize,
        sum: F,
        polys: &mut CudaViewMut<FieldBinding>,
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
        mut buf: RefMut<CudaViewMut<FieldBinding>>,
        mut round_evals: RefMut<CudaViewMut<FieldBinding>>,
    ) -> Result<(), DriverError> {
        let num_blocks_per_poly = self.max_blocks_per_sm()?;
        for k in 0..max_degree + 1 {
            let device_k = self
                .gpu
                .htod_copy(vec![F::to_montgomery_form(F::from(k as u64))])?;
            let fold_into_half = self.gpu.get_func("sumcheck", "fold_into_half").unwrap();
            let launch_config = LaunchConfig {
                grid_dim: ((num_blocks_per_poly * num_polys) as u32, 1, 1),
                block_dim: (1024, 1, 1),
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
                        &device_k,
                    ),
                )?;
            };
            let combine_and_sum = self.gpu.get_func("sumcheck", "combine_and_sum").unwrap();
            let launch_config = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                combine_and_sum.launch(
                    launch_config,
                    (
                        &mut *buf,
                        &mut *round_evals,
                        1 << (initial_poly_num_vars - round - 1),
                        num_polys,
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
        let num_blocks_per_poly = self.max_blocks_per_sm()?;
        let launch_config = LaunchConfig {
            grid_dim: ((num_blocks_per_poly * num_polys) as u32, 1, 1),
            block_dim: (1024, 1, 1),
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

    use cudarc::{driver::DriverError, nvrtc::Ptx};
    use ff::Field;
    use halo2curves::bn256::Fr;
    use itertools::Itertools;
    use rand::rngs::OsRng;

    use crate::{cpu, GPUApiWrapper, MULTILINEAR_PTX, SUMCHECK_PTX};

    #[test]
    fn test_eval_at_k_and_combine() -> Result<(), DriverError> {
        let num_vars = 20;
        let num_polys = 4;
        let max_degree = 4;
        let rng = OsRng::default();

        let combine_function = |args: &Vec<Fr>| args.iter().product();

        let polys = (0..num_polys)
            .map(|_| (0..1 << num_vars).map(|_| Fr::random(rng)).collect_vec())
            .collect_vec();

        let mut gpu_api_wrapper = GPUApiWrapper::<Fr>::setup()?;
        gpu_api_wrapper.gpu.load_ptx(
            Ptx::from_src(MULTILINEAR_PTX),
            "multilinear",
            &["convert_to_montgomery_form"],
        )?;
        gpu_api_wrapper.gpu.load_ptx(
            Ptx::from_src(SUMCHECK_PTX),
            "sumcheck",
            &["fold_into_half", "combine_and_sum"],
        )?;

        let mut cpu_round_evals = vec![];
        let now = Instant::now();
        let polys = polys.iter().map(|poly| poly.as_slice()).collect_vec();
        for k in 0..max_degree + 1 {
            cpu_round_evals.push(cpu::sumcheck::eval_at_k_and_combine(
                num_vars,
                polys.as_slice(),
                &combine_function,
                Fr::from(k),
            ));
        }
        println!(
            "Time taken to eval_at_k_and_combine on cpu: {:.2?}",
            now.elapsed()
        );

        // copy polynomials to device
        let gpu_polys = gpu_api_wrapper.copy_to_device(&polys.concat())?;
        let mut buf = gpu_api_wrapper.copy_to_device(&vec![Fr::ZERO; num_polys << num_vars])?;
        let buf_view = RefCell::new(buf.slice_mut(..));
        let mut round_evals =
            gpu_api_wrapper.copy_to_device(&vec![Fr::ZERO; max_degree as usize + 1])?;
        let round_evals_view = RefCell::new(round_evals.slice_mut(..));
        let round = 0;
        let now = Instant::now();
        gpu_api_wrapper.eval_at_k_and_combine(
            num_vars,
            round,
            max_degree as usize,
            num_polys,
            &gpu_polys.slice(..),
            buf_view.borrow_mut(),
            round_evals_view.borrow_mut(),
        )?;
        gpu_api_wrapper.gpu.synchronize()?;
        println!(
            "Time taken to eval_at_k_and_combine on gpu: {:.2?}",
            now.elapsed()
        );
        let gpu_round_evals = gpu_api_wrapper.dtoh_sync_copy(round_evals.slice(..), true)?;
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
        let num_vars = 20;
        let num_polys = 4;

        let rng = OsRng::default();
        let mut polys = (0..num_polys)
            .map(|_| (0..1 << num_vars).map(|_| Fr::random(rng)).collect_vec())
            .collect_vec();

        let mut gpu_api_wrapper = GPUApiWrapper::<Fr>::setup()?;
        gpu_api_wrapper.gpu.load_ptx(
            Ptx::from_src(MULTILINEAR_PTX),
            "multilinear",
            &["convert_to_montgomery_form"],
        )?;
        gpu_api_wrapper.gpu.load_ptx(
            Ptx::from_src(SUMCHECK_PTX),
            "sumcheck",
            &["fold_into_half_in_place"],
        )?;
        // copy polynomials to device
        let mut gpu_polys = gpu_api_wrapper.copy_to_device(&polys.concat())?;
        let challenge = Fr::random(rng);
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
        gpu_api_wrapper.gpu.synchronize()?;
        println!(
            "Time taken to fold_into_half_in_place on gpu: {:.2?}",
            now.elapsed()
        );

        let gpu_result = (0..num_polys)
            .map(|i| {
                gpu_api_wrapper.dtoh_sync_copy(
                    gpu_polys.slice(i << num_vars..(i * 2 + 1) << (num_vars - 1)),
                    true,
                )
            })
            .collect::<Result<Vec<Vec<Fr>>, _>>()?;

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
        let num_vars = 19;
        let num_polys = 9;
        let max_degree = 1;

        let rng = OsRng::default();
        let polys = (0..num_polys)
            .map(|_| (0..1 << num_vars).map(|_| Fr::random(rng)).collect_vec())
            .collect_vec();

        let mut gpu_api_wrapper = GPUApiWrapper::<Fr>::setup()?;
        gpu_api_wrapper.gpu.load_ptx(
            Ptx::from_src(MULTILINEAR_PTX),
            "multilinear",
            &["convert_to_montgomery_form"],
        )?;
        gpu_api_wrapper.gpu.load_ptx(
            Ptx::from_src(SUMCHECK_PTX),
            "sumcheck",
            &[
                "fold_into_half",
                "fold_into_half_in_place",
                "combine_and_sum",
                "squeeze_challenge",
            ],
        )?;

        let mut gpu_polys = gpu_api_wrapper.copy_to_device(&polys.concat())?;
        let sum = (0..1 << num_vars).fold(Fr::ZERO, |acc, index| {
            acc + polys.iter().map(|poly| poly[index]).sum::<Fr>()
        });
        let mut buf = gpu_api_wrapper.malloc_on_device(num_polys << (num_vars - 1))?;
        let buf_view = RefCell::new(buf.slice_mut(..));

        let mut challenges = gpu_api_wrapper.malloc_on_device(num_vars)?;
        let mut round_evals = gpu_api_wrapper.malloc_on_device(num_vars * (max_degree + 1))?;
        let round_evals_view = RefCell::new(round_evals.slice_mut(..));

        let now = Instant::now();
        gpu_api_wrapper.prove_sumcheck(
            num_vars,
            num_polys,
            max_degree,
            sum,
            &mut gpu_polys.slice_mut(..),
            buf_view,
            &mut challenges.slice_mut(..),
            round_evals_view,
        )?;
        gpu_api_wrapper.gpu.synchronize()?;
        println!(
            "Time taken to prove sumcheck on gpu : {:.2?}",
            now.elapsed()
        );

        let challenges = gpu_api_wrapper.dtoh_sync_copy(challenges.slice(..), true)?;
        let round_evals = (0..num_vars)
            .map(|i| {
                gpu_api_wrapper.dtoh_sync_copy(
                    round_evals.slice(i * (max_degree + 1)..(i + 1) * (max_degree + 1)),
                    true,
                )
            })
            .collect::<Result<Vec<Vec<Fr>>, _>>()?;
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
        Ok(())
    }
}
