// silence warnings due to bindgen
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::{marker::PhantomData, sync::Arc, time::Instant};

use cudarc::driver::{
    CudaDevice, CudaSlice, CudaView, DeviceRepr, DriverError, LaunchAsync, LaunchConfig,
};
use ff::PrimeField;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

mod cpu;
pub mod fieldbinding;
pub mod gpu;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe impl DeviceRepr for FieldBinding {}
impl Default for FieldBinding {
    fn default() -> Self {
        Self { data: 0 }
    }
}

const SUMCHECK_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sumcheck.ptx"));

/// Wrapper struct for APIs using GPU
pub struct GPUApiWrapper<F: PrimeField + From<FieldBinding> + Into<FieldBinding>> {
    gpu: Arc<CudaDevice>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + From<FieldBinding> + Into<FieldBinding>> GPUApiWrapper<F> {
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
        let device_data = self
            .gpu
            .htod_copy(host_data.into_par_iter().map(|&eval| eval.into()).collect())?;
        Ok(device_data)
    }

    pub fn malloc_on_device(&self, len: usize) -> Result<CudaSlice<FieldBinding>, DriverError> {
        let device_ptr = unsafe { self.gpu.alloc::<FieldBinding>(len)? };
        Ok(device_ptr)
    }

    pub fn dtoh_sync_copy(
        &self,
        device_data: &CudaView<FieldBinding>,
    ) -> Result<Vec<F>, DriverError> {
        let host_data = self.gpu.dtoh_sync_copy(device_data)?;
        Ok(host_data.into_iter().map(|b| b.into()).collect_vec())
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
