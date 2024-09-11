use crate::{
    fieldbinding::{FromFieldBinding, ToFieldBinding},
    FieldBinding, GPUApiWrapper, LibraryError,
};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
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
    fn get_cuda_slice(
        &mut self,
        gpu: &mut GPUApiWrapper<F>,
        count: usize,
        add_len: usize,
    ) -> Result<TranscriptInner, LibraryError>;
}

pub struct TranscriptInner {
    pub start: CudaSlice<u8>,
    pub cursor: usize,
    pub end: usize,
}

impl TranscriptInner {
    fn new(start: CudaSlice<u8>, cursor: usize, end: usize) -> Self {
        Self { start, cursor, end }
    }
}

pub struct CudaKeccakTranscript<F> {
    stream: Cursor<Vec<u8>>,
    _marker: PhantomData<F>,
    inner: Option<TranscriptInner>,
}

impl<F: PrimeField> CudaKeccakTranscript<F> {
    pub fn new() -> Self {
        Self {
            stream: Cursor::new(vec![]),
            _marker: PhantomData,
            inner: None,
        }
    }

    fn read_field_element(&mut self) -> Result<Vec<u8>, LibraryError> {
        let mut repr: Vec<u8> = vec![];
        self.stream
            .read_exact(repr.as_mut())
            .map_err(|_| LibraryError::Transcript)?;

        Ok(repr)
    }

    fn read_field_elements(&mut self, n: usize) -> Result<Vec<u8>, LibraryError> {
        let mut res = vec![];
        for _ in 0..n {
            let mut fe = self.read_field_element()?.to_vec();
            res.append(&mut fe);
        }
        Ok(res)
    }
}

impl<F: PrimeField + ToFieldBinding<F> + FromFieldBinding<F>> CudaTranscript<F>
    for CudaKeccakTranscript<F>
{
    fn get_hash_method(&self) -> Hash {
        return Hash::Keccack256;
    }

    fn get_cuda_slice(
        &mut self,
        gpu: &mut GPUApiWrapper<F>,
        count: usize,
        add_len: usize,
    ) -> Result<TranscriptInner, LibraryError> {
        let host_data = self.read_field_elements(count)?;
        let inner = gpu
            .copy_and_malloc_transcript(host_data.as_slice(), add_len)
            .map_err(|err| LibraryError::Driver(err))?;
        Ok(TranscriptInner::new(inner.0, inner.1, inner.2))
    }
}
