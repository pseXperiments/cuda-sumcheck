use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../cuda_sumcheck")
        .copy_to("./ptx_folder/path.ptx")
        .build()
        .unwrap();
}
