# Overview

This project aims to implement Sumcheck protocol on GPU via CUDA, following the algorithm described from [here](https://github.com/ingonyama-zk/super-sumcheck). This project aims to minimize C++ code and provide easy to use Rust API for launching kernel by leveraging [bindgen](https://github.com/rust-lang/rust-bindgen) and [cudarc](https://github.com/coreylowman/cudarc).

## Architecture

![cuda-sumcheck-architecture](https://github.com/pseXperiments/cuda-sumcheck/assets/59155248/d288b9d3-4fbd-4789-ba4b-de684efc3f4f)

- `bindgen`: generates Rust struct defined inside CUDA kernels written in C++
- `cudarc`: safe abstraction for interop with GPU driver

## Structure

```
.
├── Cargo.lock
├── Cargo.toml
├── build.rs
├── docs
│   └── design.md
└── src
    ├── cuda
    │   ├── includes
    │   │   ├── barretenberg
    │   │   ├── prime_field.h
    │   │   ├── test.cpp
    │   │   └── wrapper.h
    │   └── kernels
    │       ├── multilinear.cu
    │       ├── scalar_multiplication.cu
    │       └── sumcheck.cu
    ├── field.rs
    └── lib.rs
```

- `build.rs` contains the build script for building the whole project

- `src/cuda/includes/barretenberg/` contains barretenberg lib (https://github.com/AztecProtocol/barretenberg/tree/master) for field implementation in C++

- `src/cuda/includes/prime_field.h` declares `FieldBinding` struct in C representing barretenberg `field` struct. This wrapping is due to the lack of support for C++ language of `bindgen` crate

- `src/cuda/includes/wrapper.h` is interface file for `bindgen` to generate Rust struct from C struct

- In `src/cuda/includes/kernel/`, multiple kernel functions are defined

- In `field.rs`, conversion between `FieldBinding` struct and `halo2curves`' field struct is implemented

- In `lib.rs`, expose APIs to test CUDA kernels via `GPUApiWrapper` struct
