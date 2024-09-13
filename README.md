# Overview

This project aims to implement Sumcheck protocol to run on GPU via CUDA, following the algorithms described in the [paper](https://eprint.iacr.org/2024/1046.pdf).
Currently algorithm 1 is implemented and tested with bn254 scalar field. Future plan is to implement algorithm 3 and support small prime fields such as Goldilocks.

Only kernels are written in C++ and uses Rust to launch kernels and handle device datas by using [bindgen](https://github.com/rust-lang/rust-bindgen) and [cudarc](https://github.com/coreylowman/cudarc).

## Environment

The implementation has been tested on the following spec.

### OS

Ubuntu 22.04.4 LTS

### GPU

1 NVIDIA A10G Tensor GPU, 24GB Ram

### CUDA version

nvcc version: 12.1
C++ compiler version: clang-16
C++ language standard: C++20

## Architecture

![cuda-sumcheck-architecture](https://github.com/pseXperiments/cuda-sumcheck/assets/59155248/d288b9d3-4fbd-4789-ba4b-de684efc3f4f)
