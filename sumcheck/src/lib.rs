// silence warnings due to bindgen
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use cpu::parallel::parallelize;
use cudarc::driver::{
    CudaDevice, CudaSlice, CudaView, DeviceRepr, DriverError, LaunchAsync, LaunchConfig,
};
use ff::PrimeField;
use fieldbinding::{FromFieldBinding, ToFieldBinding};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

mod cpu;
pub mod fieldbinding;
pub mod gpu;
pub mod transcript;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe impl DeviceRepr for FieldBinding {}
impl Default for FieldBinding {
    fn default() -> Self {
        Self { data: [0; 4] }
    }
}

// include the compiled PTX code as string
const MULTILINEAR_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/multilinear.ptx"));
const SUMCHECK_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sumcheck.ptx"));

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

    pub fn copy_to_device(
        &mut self,
        host_data: &[F],
    ) -> Result<CudaSlice<FieldBinding>, DriverError> {
        let device_data = self.gpu.htod_copy(
            host_data
                .into_par_iter()
                .map(|&eval| F::to_canonical_form(eval))
                .collect(),
        )?;
        let convert_to_montgomery_form = self
            .gpu
            .get_func("multilinear", "convert_to_montgomery_form")
            .unwrap();
        let size = host_data.len();
        unsafe {
            convert_to_montgomery_form.launch(
                LaunchConfig::for_num_elems(size as u32),
                (&device_data, size),
            )?;
        };
        Ok(device_data)
    }

    pub fn malloc_on_device(&self, len: usize) -> Result<CudaSlice<FieldBinding>, DriverError> {
        let device_ptr = unsafe { self.gpu.alloc(len << 5)? };
        Ok(device_ptr)
    }

    pub fn copy_and_malloc_transcript(
        &mut self,
        host_data: &[u8],
        add_len: usize,
    ) -> Result<(CudaSlice<u8>, usize, usize), DriverError> {
        let mut padding = vec![0; add_len];
        let mut data = host_data.to_vec();
        data.append(&mut padding);

        let mut device_data = unsafe { self.gpu.alloc(data.len()) }?;
        self.gpu.htod_copy_into::<u8>(data, &mut device_data)?;
        let cursor = host_data.len();
        let end = cursor + add_len;
        Ok((device_data, cursor, end))
    }

    pub fn dtoh_sync_copy<T: DeviceRepr>(
        &self,
        device_data: CudaView<T>,
    ) -> Result<Vec<T>, DriverError> {
        self.gpu.dtoh_sync_copy::<T, CudaView<T>>(&device_data)
    }

    pub fn max_blocks_per_sm(&self) -> Result<usize, DriverError> {
        Ok(self.gpu.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR)? as usize)
    }

    pub fn num_sm(&self) -> Result<usize, DriverError> {
        Ok(self.gpu.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )? as usize)
    }

    pub fn max_threads_per_sm(&self) -> Result<usize, DriverError> {
        Ok(self.gpu.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)? as usize)
    }

    pub fn shared_mem_bytes_per_block(&self) -> Result<usize, DriverError> {
        Ok(self.gpu.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)? as usize)
    }
}

#[derive(Debug)]
pub enum LibraryError {
    Driver(DriverError),
    Transcript,
}
