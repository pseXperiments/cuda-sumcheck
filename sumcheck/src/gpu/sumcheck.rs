use std::cell::{RefCell, RefMut};

use cudarc::driver::{CudaSlice, CudaView, CudaViewMut, LaunchAsync, LaunchConfig};
use ff::PrimeField;

use crate::{
    fieldbinding::{FromFieldBinding, ToFieldBinding},
    transcript::Keccak256Transcript,
    FieldBinding, GPUApiWrapper, LibraryError,
};

impl<F: PrimeField + FromFieldBinding<F> + ToFieldBinding<F>> GPUApiWrapper<F> {
    pub fn prove_sumcheck(
        &self,
        num_vars: usize,
        num_polys: usize,
        max_degree: usize,
        sum: F,
        polys: &mut CudaViewMut<FieldBinding>,
        device_ks: &[CudaView<FieldBinding>],
        buf: RefCell<CudaViewMut<FieldBinding>>,
        challenge: &mut CudaSlice<FieldBinding>,
        round_evals: RefCell<CudaViewMut<FieldBinding>>,
        transcript: &mut Keccak256Transcript<F>,
        // highly temporary
        gamma: &CudaView<FieldBinding>,
    ) -> Result<(), LibraryError> {
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
                transcript,
                gamma,
            )?;
            // squeeze challenge
            let alpha = transcript.squeeze_challenge();
            let c = vec![alpha];
            self.overwrite_to_device(c.as_slice(), challenge)
                .map_err(|e| LibraryError::Driver(e))?;
            // fold_into_half_in_place
            self.fold_into_half_in_place(
                initial_poly_num_vars,
                round,
                num_polys,
                polys,
                challenge,
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
        transcript: &mut Keccak256Transcript<F>,
        gamma: &CudaView<FieldBinding>,
    ) -> Result<(), LibraryError> {
        let num_blocks_per_poly = self
            .max_blocks_per_sm()
            .map_err(|e| LibraryError::Driver(e))?
            / num_polys
            * self.num_sm().map_err(|e| LibraryError::Driver(e))?;
        let num_threads_per_block = 1024;
        for k in 0..max_degree + 1 {
            let fold_into_half = self.gpu.get_func("sumcheck", "fold_into_half").unwrap();
            let launch_config = LaunchConfig {
                grid_dim: ((num_blocks_per_poly * num_polys) as u32, 1, 1),
                block_dim: (num_threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                fold_into_half
                    .launch(
                        launch_config,
                        (
                            initial_poly_num_vars - round,
                            1 << initial_poly_num_vars,
                            num_blocks_per_poly,
                            polys,
                            &mut *buf,
                            &device_ks[k],
                        ),
                    )
                    .map_err(|e| LibraryError::Driver(e))?;
            };
            let size = 1 << (initial_poly_num_vars - round - 1);
            let combine = self.gpu.get_func("sumcheck", "combine").unwrap();
            let launch_config = LaunchConfig {
                grid_dim: (num_blocks_per_poly as u32, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                combine
                    .launch(launch_config, (&mut *buf, size, num_polys, gamma))
                    .map_err(|e| LibraryError::Driver(e))?;
            };
            let sum = self.gpu.get_func("sumcheck", "sum").unwrap();
            let launch_config = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (num_threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                sum.launch(launch_config, (&mut *buf, &mut *round_evals, size >> 1, k))
                    .map_err(|e| LibraryError::Driver(e))?;
            };
        }
        let round_eval_values = self
            .dtoh_sync_copy(&round_evals.slice(0..(max_degree + 1) as usize), true)
            .map_err(|e| LibraryError::Driver(e))?;
        transcript.write_field_elements(&round_eval_values)?;
        Ok(())
    }

    pub(crate) fn fold_into_half_in_place(
        &self,
        initial_poly_num_vars: usize,
        round: usize,
        num_polys: usize,
        polys: &mut CudaViewMut<FieldBinding>,
        challenge: &CudaSlice<FieldBinding>,
    ) -> Result<(), LibraryError> {
        let fold_into_half_in_place = self
            .gpu
            .get_func("sumcheck", "fold_into_half_in_place")
            .unwrap();
        let num_blocks_per_poly = self
            .max_blocks_per_sm()
            .map_err(|e| LibraryError::Driver(e))?
            / num_polys
            * self.num_sm().map_err(|e| LibraryError::Driver(e))?;
        let num_threads_per_block = 1024;
        let launch_config = LaunchConfig {
            grid_dim: ((num_blocks_per_poly * num_polys) as u32, 1, 1),
            block_dim: (num_threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            fold_into_half_in_place
                .launch(
                    launch_config,
                    (
                        initial_poly_num_vars - round,
                        1 << initial_poly_num_vars,
                        num_blocks_per_poly,
                        polys,
                        challenge,
                    ),
                )
                .map_err(|e| LibraryError::Driver(e))?;
        };
        Ok(())
    }
}

// #[cfg(test)]
// mod tests {
//     use std::{cell::RefCell, time::Instant};

//     use cudarc::nvrtc::Ptx;
//     use ff::Field;
//     use halo2curves::bn256::Fr;
//     use itertools::Itertools;
//     use rand::rngs::OsRng;

//     use crate::{
//         cpu, fieldbinding::ToFieldBinding, transcript::Keccak256Transcript, GPUApiWrapper,
//         LibraryError, MULTILINEAR_PTX, SUMCHECK_PTX,
//     };

//     #[test]
//     fn test_eval_at_k_and_combine() -> Result<(), LibraryError> {
//         let num_vars = 10;
//         let num_polys = 3;
//         let max_degree = 3;
//         let rng = OsRng::default();

//         let combine_function = |args: &Vec<Fr>| args.iter().product();

//         let polys = (0..num_polys)
//             .map(|_| {
//                 (0..1 << num_vars)
//                     .map(|i| {
//                         if i < 1024 {
//                             Fr::random(rng)
//                         } else {
//                             Fr::from(i)
//                         }
//                     })
//                     .collect_vec()
//             })
//             .collect_vec();

//         let mut gpu_api_wrapper =
//             GPUApiWrapper::<Fr>::setup().map_err(|e| LibraryError::Driver(e))?;
//         gpu_api_wrapper
//             .gpu
//             .load_ptx(
//                 Ptx::from_src(MULTILINEAR_PTX),
//                 "multilinear",
//                 &["convert_to_montgomery_form"],
//             )
//             .map_err(|e| LibraryError::Driver(e))?;
//         gpu_api_wrapper
//             .gpu
//             .load_ptx(
//                 Ptx::from_src(SUMCHECK_PTX),
//                 "sumcheck",
//                 &["fold_into_half", "combine", "sum"],
//             )
//             .map_err(|e| LibraryError::Driver(e))?;

//         let mut cpu_round_evals = vec![];
//         let now = Instant::now();
//         let polys = polys.iter().map(|poly| poly.as_slice()).collect_vec();
//         for k in 0..max_degree + 1 {
//             cpu_round_evals.push(cpu::sumcheck::eval_at_k_and_combine(
//                 num_vars,
//                 polys.as_slice(),
//                 &combine_function,
//                 Fr::from(k),
//             ));
//         }
//         println!(
//             "Time taken to eval_at_k_and_combine on cpu: {:.2?}",
//             now.elapsed()
//         );

//         // copy polynomials to device
//         let gpu_polys = gpu_api_wrapper
//             .copy_to_device(&polys.concat())
//             .map_err(|e| LibraryError::Driver(e))?;
//         let device_ks = (0..max_degree + 1)
//             .map(|k| {
//                 gpu_api_wrapper
//                     .gpu
//                     .htod_copy(vec![Fr::to_montgomery_form(Fr::from(k as u64))])
//             })
//             .collect::<Result<Vec<_>, _>>()
//             .map_err(|e| LibraryError::Driver(e))?;
//         let mut buf = gpu_api_wrapper
//             .malloc_on_device(num_polys << (num_vars - 1))
//             .map_err(|e| LibraryError::Driver(e))?;
//         let buf_view = RefCell::new(buf.slice_mut(..));
//         let mut round_evals = gpu_api_wrapper
//             .malloc_on_device(max_degree as usize + 1)
//             .map_err(|e| LibraryError::Driver(e))?;
//         let round_evals_view = RefCell::new(round_evals.slice_mut(..));
//         let round = 0;
//         let mut transcript = Keccak256Transcript::<Fr>::new();
//         let now = Instant::now();
//         gpu_api_wrapper.eval_at_k_and_combine(
//             num_vars,
//             round,
//             max_degree as usize,
//             num_polys,
//             &gpu_polys.slice(..),
//             &device_ks
//                 .iter()
//                 .map(|device_k| device_k.slice(..))
//                 .collect_vec(),
//             buf_view.borrow_mut(),
//             round_evals_view.borrow_mut(),
//             &mut transcript,
//         )?;
//         println!(
//             "Time taken to eval_at_k_and_combine on gpu: {:.2?}",
//             now.elapsed()
//         );
//         let gpu_round_evals = gpu_api_wrapper
//             .dtoh_sync_copy(&round_evals.slice(0..(max_degree + 1) as usize), true)
//             .map_err(|e| LibraryError::Driver(e))?;
//         cpu_round_evals
//             .iter()
//             .zip_eq(gpu_round_evals.iter())
//             .for_each(|(a, b)| {
//                 assert_eq!(a, b);
//             });

//         Ok(())
//     }

//     #[test]
//     fn test_fold_into_half_in_place() -> Result<(), LibraryError> {
//         let num_vars = 6;
//         let num_polys = 4;

//         let rng = OsRng::default();
//         let mut polys = (0..num_polys)
//             .map(|_| {
//                 (0..1 << num_vars)
//                     .map(|i| {
//                         if i < 1024 {
//                             Fr::random(rng)
//                         } else {
//                             Fr::from(i)
//                         }
//                     })
//                     .collect_vec()
//             })
//             .collect_vec();

//         let mut gpu_api_wrapper =
//             GPUApiWrapper::<Fr>::setup().map_err(|e| LibraryError::Driver(e))?;
//         gpu_api_wrapper
//             .gpu
//             .load_ptx(
//                 Ptx::from_src(MULTILINEAR_PTX),
//                 "multilinear",
//                 &["convert_to_montgomery_form"],
//             )
//             .map_err(|e| LibraryError::Driver(e))?;
//         gpu_api_wrapper
//             .gpu
//             .load_ptx(
//                 Ptx::from_src(SUMCHECK_PTX),
//                 "sumcheck",
//                 &["fold_into_half_in_place"],
//             )
//             .map_err(|e| LibraryError::Driver(e))?;
//         // copy polynomials to device
//         let mut gpu_polys = gpu_api_wrapper
//             .copy_to_device(&polys.concat())
//             .map_err(|e| LibraryError::Driver(e))?;
//         let challenge = Fr::random(rng);

//         let gpu_challenge = gpu_api_wrapper
//             .copy_to_device(&vec![challenge])
//             .map_err(|e| LibraryError::Driver(e))?;
//         let round = 0;

//         let now = Instant::now();
//         gpu_api_wrapper.fold_into_half_in_place(
//             num_vars,
//             round,
//             num_polys,
//             &mut gpu_polys.slice_mut(..),
//             &gpu_challenge,
//         )?;
//         println!(
//             "Time taken to fold_into_half_in_place on gpu: {:.2?}",
//             now.elapsed()
//         );

//         let gpu_result = (0..num_polys)
//             .map(|i| {
//                 gpu_api_wrapper.dtoh_sync_copy(
//                     &gpu_polys.slice(i << num_vars..(i * 2 + 1) << (num_vars - 1)),
//                     true,
//                 )
//             })
//             .collect::<Result<Vec<Vec<Fr>>, _>>()
//             .map_err(|e| LibraryError::Driver(e))?;

//         let now = Instant::now();
//         (0..num_polys)
//             .for_each(|i| cpu::sumcheck::fold_into_half_in_place(&mut polys[i], challenge));
//         println!("Time taken to fold_into_half on cpu: {:.2?}", now.elapsed());
//         polys
//             .iter_mut()
//             .for_each(|poly| poly.truncate(1 << (num_vars - 1)));

//         gpu_result
//             .into_iter()
//             .zip_eq(polys)
//             .for_each(|(gpu_result, cpu_result)| {
//                 assert_eq!(gpu_result, cpu_result);
//             });

//         Ok(())
//     }

//     #[test]
//     fn test_prove_sumcheck() -> Result<(), LibraryError> {
//         let num_vars = 25;
//         let num_polys = 2;
//         let max_degree = 2;

//         let rng = OsRng::default();
//         let polys = (0..num_polys)
//             .map(|_| {
//                 (0..1 << num_vars)
//                     .map(|i| {
//                         if i < 1024 {
//                             Fr::random(rng)
//                         } else {
//                             Fr::from(i)
//                         }
//                     })
//                     .collect_vec()
//             })
//             .collect_vec();

//         let mut gpu_api_wrapper =
//             GPUApiWrapper::<Fr>::setup().map_err(|e| LibraryError::Driver(e))?;
//         gpu_api_wrapper
//             .gpu
//             .load_ptx(
//                 Ptx::from_src(MULTILINEAR_PTX),
//                 "multilinear",
//                 &["convert_to_montgomery_form"],
//             )
//             .map_err(|e| LibraryError::Driver(e))?;
//         gpu_api_wrapper
//             .gpu
//             .load_ptx(
//                 Ptx::from_src(SUMCHECK_PTX),
//                 "sumcheck",
//                 &[
//                     "fold_into_half",
//                     "fold_into_half_in_place",
//                     "combine",
//                     "sum",
//                 ],
//             )
//             .map_err(|e| LibraryError::Driver(e))?;
//         let mut transcript = Keccak256Transcript::<Fr>::new();
//         let now = Instant::now();
//         let mut gpu_polys = gpu_api_wrapper
//             .copy_to_device(&polys.concat())
//             .map_err(|e| LibraryError::Driver(e))?;
//         let sum = (0..1 << num_vars).fold(Fr::ZERO, |acc, index| {
//             acc + polys.iter().map(|poly| poly[index]).product::<Fr>()
//         });
//         let device_ks = (0..max_degree + 1)
//             .map(|k| {
//                 gpu_api_wrapper
//                     .gpu
//                     .htod_copy(vec![Fr::to_montgomery_form(Fr::from(k as u64))])
//             })
//             .collect::<Result<Vec<_>, _>>()
//             .map_err(|e| LibraryError::Driver(e))?;
//         let mut buf = gpu_api_wrapper
//             .malloc_on_device(num_polys << (num_vars - 1))
//             .map_err(|e| LibraryError::Driver(e))?;
//         let buf_view = RefCell::new(buf.slice_mut(..));

//         let mut challenges = gpu_api_wrapper
//             .malloc_on_device(1)
//             .map_err(|e| LibraryError::Driver(e))?;
//         let mut round_evals = gpu_api_wrapper
//             .malloc_on_device(max_degree + 1)
//             .map_err(|e| LibraryError::Driver(e))?;
//         let round_evals_view = RefCell::new(round_evals.slice_mut(..));
//         println!("Time taken to copy data to device : {:.2?}", now.elapsed());

//         let now = Instant::now();
//         gpu_api_wrapper.prove_sumcheck(
//             num_vars,
//             num_polys,
//             max_degree,
//             sum,
//             &mut gpu_polys.slice_mut(..),
//             &device_ks
//                 .iter()
//                 .map(|device_k| device_k.slice(..))
//                 .collect_vec(),
//             buf_view,
//             &mut challenges,
//             round_evals_view,
//             &mut transcript,
//         )?;
//         gpu_api_wrapper
//             .gpu
//             .synchronize()
//             .map_err(|e| LibraryError::Driver(e))?;
//         println!(
//             "Time taken to prove sumcheck on gpu : {:.2?}",
//             now.elapsed()
//         );

//         let proof = transcript.into_proof();
//         let mut transcript = Keccak256Transcript::<Fr>::from_proof(&proof);
//         let result = cpu::sumcheck::verify_sumcheck(num_vars, max_degree, sum, &mut transcript);
//         assert!(result);
//         Ok(())
//     }
// }
