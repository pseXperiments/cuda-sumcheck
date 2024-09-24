use crate::{
    fieldbinding::{FromFieldBinding, ToFieldBinding},
    GPUApiWrapper, LibraryError,
};
use cudarc::driver::CudaSlice;
use ff::PrimeField;
use halo2curves::serde::SerdeObject;
use itertools::Itertools;
use std::io::{Cursor, Read, Write};

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

pub fn from_u8_to_f<F: PrimeField + SerdeObject>(v: Vec<u8>) -> Vec<F> {
    let src: Vec<&[u8]> = v.chunks(32).collect();
    src.into_iter()
        .map(|l| {
            let data = l.chunks(8).collect_vec();
            F::from_raw_bytes_unchecked(data.concat().as_slice()) * F::ONE
        })
        .collect_vec()
}

pub struct CudaKeccakTranscript<F> {
    pub stream: Cursor<Vec<u8>>,
    pub state: F,
    pub inner: Option<TranscriptInner>,
}

impl<F: PrimeField + SerdeObject> CudaKeccakTranscript<F> {
    pub fn new(state: &F) -> Self {
        Self {
            stream: Cursor::new(vec![]),
            state: state.clone(),
            inner: None,
        }
    }

    pub fn squeeze_challenge(&self) -> F {
        self.state.clone()
    }

    fn write_field_element(&mut self, fe: &F) -> Result<(), LibraryError> {
        let mut repr = fe.to_repr();
        repr.as_mut();
        self.stream
            .write_all(repr.as_ref())
            .map_err(|_| LibraryError::Transcript)
    }

    pub fn from_vec(v: Vec<F>) -> Result<Self, LibraryError> {
        let mut new_t = Self::new(&v[v.len() - 1]);
        for fe in v.iter() {
            new_t.write_field_element(fe)?;
        }
        Ok(new_t)
    }

    pub fn read_field_element(&mut self) -> Result<Vec<u8>, LibraryError> {
        let mut repr: Vec<u8> = vec![];
        self.stream
            .read_exact(repr.as_mut())
            .map_err(|_| LibraryError::Transcript)?;
        
        self.state = from_u8_to_f::<F>(repr.clone())[0];
        Ok(repr)
    }

    pub fn read_field_elements(&mut self, n: usize) -> Result<Vec<u8>, LibraryError> {
        let mut res = vec![];
        for _ in 0..n {
            let mut fe = self.read_field_element()?.to_vec();
            res.append(&mut fe);
        }
        Ok(res)
    }
}

impl<F: PrimeField + ToFieldBinding<F> + FromFieldBinding<F> + SerdeObject> CudaTranscript<F>
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
