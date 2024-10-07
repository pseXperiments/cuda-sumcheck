## Setup CUDA development environment

### First option: using Rust-CUDA (failed)

1. Install Rust
2. Install Rust-CUDA
    - `error: failed to run custom build command for rustc_codegen_nvvm v0.3.0`
    - Okay, llvm-7 package is not supported on Ubuntu 22.04
    - Just found out this project is not maintained over 2 years, and build fails

### Second option: using cudarc

https://github.com/coreylowman/cudarc?tab=readme-ov-file
- Provides safe API for CUDA functions
- Example usage:
    - it seems we write kernel in C++ and pass the program as string literal to `PTX_SRC`
    - compile the program to PTX and then run
```rust
use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

const PTX_SRC: &str = "
extern \"C\" __global__ void matmul(float* A, float* B, float* C, int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    // printf(\"pos, (%d, %d) - N %d - value %d\\n\", ROW, COL, N, tmpSum);
    C[ROW * N + COL] = tmpSum;
}
";

fn main() -> Result<(), DriverError> {
    let start = std::time::Instant::now();

    let ptx = compile_ptx(PTX_SRC).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());

    let dev = CudaDevice::new(0)?;
    println!("Built in {:?}", start.elapsed());

    dev.load_ptx(ptx, "matmul", &["matmul"])?;
    let f = dev.get_func("matmul", "matmul").unwrap();
    println!("Loaded in {:?}", start.elapsed());

    let a_host = [1.0f32, 2.0, 3.0, 4.0];
    let b_host = [1.0f32, 2.0, 3.0, 4.0];
    let mut c_host = [0.0f32; 4];

    let a_dev = dev.htod_sync_copy(&a_host)?;
    let b_dev = dev.htod_sync_copy(&b_host)?;
    let mut c_dev = dev.htod_sync_copy(&c_host)?;

    println!("Copied in {:?}", start.elapsed());

    let cfg = LaunchConfig {
        block_dim: (2, 2, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, 2i32)) }?;

    dev.dtoh_sync_copy_into(&c_dev, &mut c_host)?;
    println!("Found {:?} in {:?}", c_host, start.elapsed());
    Ok(())
}
```

### Third option: C++ programming in CUDA and use inside Rust

- Writing kernel in original CUDA toolchain
- Difficulty in interop between Rust and C++
    - it seems `cudarc` can handle this problem, however if it isn't the case, still using `cudarc` is better option than programming the interop from scratch

## Conclusion

- Write kernel in C++
- Use good Rust wrapper for kernel functions (`cudarc` is viable option for now)
