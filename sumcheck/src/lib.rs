use std::time::Instant;
use cudarc::driver::{CudaDevice, LaunchConfig, DeviceRepr, DriverError, LaunchAsync};
use cudarc::nvrtc::Ptx;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// TODO : Replace this with our sumcheck struct
unsafe impl DeviceRepr for MyStruct {}
impl Default for MyStruct {
    fn default() -> Self{
        Self{ data: [0.0; 4]}
    }
}

// include the compiled PTX code as string
const CUDA_KERNEL_MY_STRUCT: &str = include_str!(concat!(env!("OUT_DIR"), "/sumcheck.ptx"));

fn main() -> Result<(), DriverError> {
    // setup GPU device
    let now = Instant::now();

    let gpu = CudaDevice::new(0)?;

    println!("Time taken to initialise CUDA: {:.2?}", now.elapsed());

    // compile ptx
    let now = Instant::now();

    let ptx = Ptx::from_src(CUDA_KERNEL_MY_STRUCT);
    gpu.load_ptx(ptx, "my_module", &["my_struct_kernel"])?;

    println!("Time taken to compile and load PTX: {:.2?}", now.elapsed());

    // create data
    let now = Instant::now();

    let n = 10_usize;
    let my_structs = vec![MyStruct { data: [1.0; 4] }; n];

    // copy to GPU
    let gpu_my_structs = gpu.htod_copy(my_structs)?;

    println!("Time taken to initialise data: {:.2?}", now.elapsed());

    let now = Instant::now();

    let f = gpu.get_func("my_module", "my_struct_kernel").unwrap();

    unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (&gpu_my_structs, n)) }?;

    println!("Time taken to call kernel: {:.2?}", now.elapsed());

    let my_structs = gpu.sync_reclaim(gpu_my_structs)?;

    assert!(my_structs.iter().all(|i| i.data == [2.0; 4]));

    Ok(())
}
