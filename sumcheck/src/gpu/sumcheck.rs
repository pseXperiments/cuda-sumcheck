use std::cell::{RefCell, RefMut};

use cudarc::driver::{CudaView, CudaViewMut, DriverError, LaunchAsync, LaunchConfig};
use ff::PrimeField;

use crate::{
    fieldbinding::{FromFieldBinding, ToFieldBinding},
    transcript::TranscriptInner,
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
        transcript: &mut TranscriptInner,
        transcript_state: RefCell<CudaViewMut<FieldBinding>>,
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
                transcript,
                transcript_state.borrow_mut(),
            )?;
            // fold_into_half_in_place
            self.fold_into_half_in_place(
                initial_poly_num_vars,
                round,
                num_polys,
                polys,
                transcript,
                transcript_state.borrow_mut(),
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
        transcript: &mut TranscriptInner,
        mut transcript_state: RefMut<CudaViewMut<FieldBinding>>,
    ) -> Result<(), DriverError> {
        let num_blocks_per_poly = self.max_blocks_per_sm()? / num_polys * self.num_sm()?;
        let num_threads_per_block = 1024;
        for k in 0..max_degree + 1 {
            let device_k = self
                .gpu
                .htod_copy(vec![F::to_montgomery_form(F::from(k as u64))])?;
            let fold_into_half = self.gpu.get_func("sumcheck", "fold_into_half").unwrap();
            let launch_config = LaunchConfig {
                grid_dim: ((num_blocks_per_poly * num_polys) as u32, 1, 1),
                block_dim: (num_threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let cursor_idx = transcript.cursor;
            let start = &mut transcript.start;
            let (mut start_view, mut cursor) = start.split_at_mut(cursor_idx);
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
                        size >> 1,
                        round * (max_degree + 1) + k,
                        &mut start_view,
                        &mut cursor,
                        &mut *transcript_state,
                    ),
                )?;
            };
            transcript.cursor += 32;
        }
        Ok(())
    }

    pub(crate) fn fold_into_half_in_place(
        &self,
        initial_poly_num_vars: usize,
        round: usize,
        num_polys: usize,
        polys: &mut CudaViewMut<FieldBinding>,
        transcript: &mut TranscriptInner,
        mut state: RefMut<CudaViewMut<FieldBinding>>,
    ) -> Result<(), DriverError> {
        let fold_into_half_in_place = self
            .gpu
            .get_func("sumcheck", "fold_into_half_in_place")
            .unwrap();
        let num_blocks_per_poly = self.max_blocks_per_sm()? / num_polys * self.num_sm()?;
        let num_threads_per_block = 1024;
        let launch_config = LaunchConfig {
            grid_dim: ((num_blocks_per_poly * num_polys) as u32, 1, 1),
            block_dim: (num_threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let cursor_idx = transcript.cursor;
        let start = &mut transcript.start;
        let (mut start_view, mut cursor) = start.split_at_mut(cursor_idx);
        unsafe {
            fold_into_half_in_place.launch(
                launch_config,
                (
                    initial_poly_num_vars - round,
                    1 << initial_poly_num_vars,
                    num_blocks_per_poly,
                    polys,
                    &mut start_view,
                    &mut cursor,
                    &mut *state,
                ),
            )?;
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, time::Instant};
    use cudarc::nvrtc::Ptx;
    use ff::Field;
    use halo2curves::{bn256::Fr, serde::SerdeObject};
    use itertools::Itertools;
    use rand::rngs::OsRng;

    use crate::{
        cpu,
        fieldbinding::FromFieldBinding,
        transcript::{CudaKeccakTranscript, CudaTranscript},
        GPUApiWrapper, LibraryError, MULTILINEAR_PTX, SUMCHECK_PTX,
    };

    fn from_u8_to_fr(v: Vec<u8>) -> Vec<Fr> {
        let src: Vec<&[u8]> = v.chunks(32).collect();
        src.into_iter()
            .map(|l| {
                let data = l.chunks(8).collect_vec();
                Fr::from_raw_bytes_unchecked(data.concat().as_slice()) * Fr::one()
            })
            .collect_vec()
    }

    #[test]
    fn test_eval_at_k_and_combine() -> Result<(), LibraryError> {
        let num_vars = 10;
        let num_polys = 4;
        let max_degree = 4;
        let rng = OsRng::default();

        let combine_function = |args: &Vec<Fr>| args.iter().product();

        let polys = (0..num_polys)
            .map(|_| (0..1 << num_vars).map(|_| Fr::random(rng)).collect_vec())
            .collect_vec();

        let mut gpu_api_wrapper =
            GPUApiWrapper::<Fr>::setup().map_err(|err| LibraryError::Driver(err))?;
        gpu_api_wrapper
            .gpu
            .load_ptx(
                Ptx::from_src(MULTILINEAR_PTX),
                "multilinear",
                &["convert_to_montgomery_form"],
            )
            .map_err(|err| LibraryError::Driver(err))?;
        gpu_api_wrapper
            .gpu
            .load_ptx(
                Ptx::from_src(SUMCHECK_PTX),
                "sumcheck",
                &["fold_into_half", "combine", "sum"],
            )
            .map_err(|err| LibraryError::Driver(err))?;

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
        let gpu_polys = gpu_api_wrapper
            .copy_to_device(&polys.concat())
            .map_err(|err| LibraryError::Driver(err))?;
        let mut buf = gpu_api_wrapper
            .malloc_on_device(num_polys << (num_vars - 1))
            .map_err(|err| LibraryError::Driver(err))?;
        let buf_view = RefCell::new(buf.slice_mut(..));
        let mut round_evals = gpu_api_wrapper
            .malloc_on_device(max_degree as usize + 1)
            .map_err(|err| LibraryError::Driver(err))?;
        let _round_evals_view = RefCell::new(round_evals.slice_mut(..));

        let count = 0; // TODO
        let add_len = (max_degree as usize + 1) * 32;
        let mut transcript = CudaKeccakTranscript::<Fr>::new(&Fr::zero());
        let mut transcript_inner =
            transcript.get_cuda_slice(&mut gpu_api_wrapper, count, add_len)?;

        let state = vec![transcript.state];
        let mut state_slice = gpu_api_wrapper
            .copy_to_device(&state.as_slice())
            .map_err(|err| LibraryError::Driver(err))?;
        let state_view = RefCell::new(state_slice.slice_mut(..));
        let round = 0;
        let now = Instant::now();
        gpu_api_wrapper
            .eval_at_k_and_combine(
                num_vars,
                round,
                max_degree as usize,
                num_polys,
                &gpu_polys.slice(..),
                buf_view.borrow_mut(),
                &mut transcript_inner,
                state_view.borrow_mut(),
            )
            .map_err(|err| LibraryError::Driver(err))?;
        gpu_api_wrapper
            .gpu
            .synchronize()
            .map_err(|err| LibraryError::Driver(err))?;
        println!(
            "Time taken to eval_at_k_and_combine on gpu: {:.2?}",
            now.elapsed()
        );
        let transcript_device = gpu_api_wrapper
            .dtoh_sync_copy(
                transcript_inner
                    .start
                    .slice(0..((max_degree + 1) as usize * 32)),
            )
            .map_err(|err| LibraryError::Driver(err))?;
        let gpu_round_evals = from_u8_to_fr(transcript_device);

        cpu_round_evals
            .iter()
            .zip_eq(gpu_round_evals.iter())
            .for_each(|(a, b)| {
                assert_eq!(a, b);
            });

        Ok(())
    }

    #[test]
    fn test_fold_into_half_in_place() -> Result<(), LibraryError> {
        let num_vars = 6;
        let num_polys = 4;

        let rng = OsRng::default();
        let mut polys = (0..num_polys)
            .map(|_| (0..1 << num_vars).map(|_| Fr::random(rng)).collect_vec())
            .collect_vec();

        let mut gpu_api_wrapper =
            GPUApiWrapper::<Fr>::setup().map_err(|err| LibraryError::Driver(err))?;
        gpu_api_wrapper
            .gpu
            .load_ptx(
                Ptx::from_src(MULTILINEAR_PTX),
                "multilinear",
                &["convert_to_montgomery_form"],
            )
            .map_err(|err| LibraryError::Driver(err))?;
        gpu_api_wrapper
            .gpu
            .load_ptx(
                Ptx::from_src(SUMCHECK_PTX),
                "sumcheck",
                &["fold_into_half_in_place"],
            )
            .map_err(|err| LibraryError::Driver(err))?;
        // copy polynomials to device
        let mut gpu_polys = gpu_api_wrapper
            .copy_to_device(&polys.concat())
            .map_err(|err| LibraryError::Driver(err))?;
        let challenge = Fr::random(rng);
        let mut gpu_challenge = gpu_api_wrapper
            .copy_to_device(&vec![challenge])
            .map_err(|err| LibraryError::Driver(err))?;
        let challenge_view = RefCell::new(gpu_challenge.slice_mut(..));
        let round = 0;

        let count = 0;
        let add_len = 0;
        let mut transcript = CudaKeccakTranscript::<Fr>::new(&challenge);
        let mut transcript_inner =
            transcript.get_cuda_slice(&mut gpu_api_wrapper, count, add_len)?;

        let now = Instant::now();
        gpu_api_wrapper
            .fold_into_half_in_place(
                num_vars,
                round,
                num_polys,
                &mut gpu_polys.slice_mut(..),
                &mut transcript_inner,
                challenge_view.borrow_mut(),
            )
            .map_err(|err| LibraryError::Driver(err))?;
        gpu_api_wrapper
            .gpu
            .synchronize()
            .map_err(|err| LibraryError::Driver(err))?;
        println!(
            "Time taken to fold_into_half_in_place on gpu: {:.2?}",
            now.elapsed()
        );

        let gpu_result = (0..num_polys)
            .map(|i| {
                gpu_api_wrapper
                    .dtoh_sync_copy(gpu_polys.slice(i << num_vars..(i * 2 + 1) << (num_vars - 1)))
                    .map(|v| v.iter().map(|b| Fr::from_montgomery_form(*b)).collect())
                    .map_err(|err| LibraryError::Driver(err))
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
    fn test_prove_sumcheck() -> Result<(), LibraryError> {
        let num_vars = 12;
        let num_polys = 4;
        let max_degree = 4;

        let rng = OsRng::default();
        let polys = (0..num_polys)
            .map(|_| (0..1 << num_vars).map(|_| Fr::random(rng)).collect_vec())
            .collect_vec();

        let mut gpu_api_wrapper =
            GPUApiWrapper::<Fr>::setup().map_err(|err| LibraryError::Driver(err))?;
        gpu_api_wrapper
            .gpu
            .load_ptx(
                Ptx::from_src(MULTILINEAR_PTX),
                "multilinear",
                &["convert_to_montgomery_form"],
            )
            .map_err(|err| LibraryError::Driver(err))?;
        gpu_api_wrapper
            .gpu
            .load_ptx(
                Ptx::from_src(SUMCHECK_PTX),
                "sumcheck",
                &[
                    "fold_into_half",
                    "fold_into_half_in_place",
                    "combine",
                    "sum",
                ],
            )
            .map_err(|err| LibraryError::Driver(err))?;

        let now = Instant::now();
        let mut gpu_polys = gpu_api_wrapper
            .copy_to_device(&polys.concat())
            .map_err(|err| LibraryError::Driver(err))?;
        let sum = (0..1 << num_vars).fold(Fr::ZERO, |acc, index| {
            acc + polys.iter().map(|poly| poly[index]).product::<Fr>()
        });
        let mut buf = gpu_api_wrapper
            .malloc_on_device(num_polys << (num_vars - 1))
            .map_err(|err| LibraryError::Driver(err))?;
        let buf_view = RefCell::new(buf.slice_mut(..));

        let mut round_evals = gpu_api_wrapper
            .malloc_on_device(num_vars * (max_degree + 1))
            .map_err(|err| LibraryError::Driver(err))?;
        let _round_evals_view = RefCell::new(round_evals.slice_mut(..));
        println!("Time taken to copy data to device : {:.2?}", now.elapsed());

        let count = 0;
        let add_len = num_vars * (max_degree + 1) * 32;
        let challenge = vec![Fr::zero()];
        let mut transcript = CudaKeccakTranscript::<Fr>::new(&Fr::zero());
        let mut gpu_challenge = gpu_api_wrapper
            .copy_to_device(&challenge)
            .map_err(|err| LibraryError::Driver(err))?;
        let challenge_view = RefCell::new(gpu_challenge.slice_mut(..));

        let mut transcript_inner =
            transcript.get_cuda_slice(&mut gpu_api_wrapper, count, add_len)?;

        let now = Instant::now();
        gpu_api_wrapper
            .prove_sumcheck(
                num_vars,
                num_polys,
                max_degree,
                sum,
                &mut gpu_polys.slice_mut(..),
                buf_view,
                &mut transcript_inner,
                challenge_view,
            )
            .map_err(|err| LibraryError::Driver(err))?;
        gpu_api_wrapper
            .gpu
            .synchronize()
            .map_err(|err| LibraryError::Driver(err))?;
        println!(
            "Time taken to prove sumcheck on gpu : {:.2?}",
            now.elapsed()
        );
        let transcript_device = gpu_api_wrapper
            .dtoh_sync_copy(
                transcript_inner
                    .start
                    .slice(0..((max_degree + 1) as usize * 32 * num_vars)),
            )
            .map_err(|err| LibraryError::Driver(err))?;

        let result = cpu::sumcheck::verify_sumcheck_transcript(
            num_vars,
            max_degree,
            sum,
            transcript_device,
        );
        assert!(result);
        Ok(())
    }
}
