use crate::{
    fieldbinding::{FromFieldBinding, ToFieldBinding},
    FieldBinding, GPUApiWrapper, LibraryError,
};
use cudarc::driver::CudaSlice;
use ff::PrimeField;
use std::{
    io::{Cursor, Read},
    marker::PhantomData,
};

pub enum Hash {
    Keccack256,
}

pub trait CudaTranscript<F: PrimeField + ToFieldBinding<F> + FromFieldBinding<F>> {
    fn get_hash_method(&self) -> Hash;
    fn get_round_evals(&self) -> Vec<F>;
    fn get_cuda_slice(
        &mut self,
        gpu: &mut GPUApiWrapper<F>,
        count: usize,
        add_len: usize,
    ) -> Result<CudaSlice<FieldBinding>, LibraryError>;
}

pub struct CudaKeccakTranscript<F> {
    stream: Cursor<Vec<u8>>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> CudaKeccakTranscript<F> {
    fn read_field_element(&mut self) -> Result<F, LibraryError> {
        let mut repr = F::Repr::default();
        self.stream
            .read_exact(repr.as_mut())
            .map_err(|_| LibraryError::Transcript)?;
        let fe = F::from_repr_vartime(repr).ok_or_else(|| LibraryError::Transcript)?;
        Ok(fe)
    }

    fn read_field_elements(&mut self, n: usize) -> Result<Vec<F>, LibraryError> {
        (0..n).map(|_| self.read_field_element()).collect()
    }
}

impl<F: PrimeField + ToFieldBinding<F> + FromFieldBinding<F>> CudaTranscript<F>
    for CudaKeccakTranscript<F>
{
    fn get_hash_method(&self) -> Hash {
        return Hash::Keccack256;
    }

    fn get_round_evals(&self) -> Vec<F> {
        vec![]
    }

    fn get_cuda_slice(
        &mut self,
        gpu: &mut GPUApiWrapper<F>,
        count: usize,
        add_len: usize,
    ) -> Result<CudaSlice<FieldBinding>, LibraryError> {
        let host_data = self.read_field_elements(count)?;
        gpu.copy_and_malloc(host_data.as_slice(), add_len)
            .map_err(|err| LibraryError::Driver(err))
    }
}
