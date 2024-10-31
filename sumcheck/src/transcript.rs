use crate::{
    fieldbinding::{FromFieldBinding, ToFieldBinding},
    utils::{arithmetic::fe_mod_from_le_bytes, hash::Hash},
    GPUApiWrapper, LibraryError,
};
use ff::PrimeField;
use halo2curves::serde::SerdeObject;
use sha3::Keccak256;
use std::{
    io::{Cursor, Read, Write},
    marker::PhantomData,
};

pub type Keccak256Transcript<F> = CudaTranscript<Keccak256, F>;

#[derive(Debug, Default)]
pub struct CudaTranscript<H, F> {
    pub stream: Cursor<Vec<u8>>,
    pub state: H,
    marker: PhantomData<F>,
}

impl<H: Hash, F: PrimeField> CudaTranscript<H, F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn into_proof(self) -> Vec<u8> {
        self.stream.into_inner()
    }

    pub fn from_proof(proof: &[u8]) -> Self {
        Self {
            state: H::default(),
            stream: Cursor::new(proof.to_vec()),
            marker: PhantomData::default(),
        }
    }

    pub fn squeeze_challenge(&mut self) -> F {
        let hash = self.state.finalize_fixed_reset();
        self.state.update(&hash);
        fe_mod_from_le_bytes(hash)
    }

    fn common_field_element(&mut self, fe: &F) -> Result<(), LibraryError> {
        self.state.update_field_element(fe);
        Ok(())
    }

    fn write_field_element(&mut self, fe: &F) -> Result<(), LibraryError> {
        self.common_field_element(fe)?;
        let mut repr = fe.to_repr();
        repr.as_mut().reverse();
        self.stream
            .write_all(repr.as_ref())
            .map_err(|err| LibraryError::Transcript(err.kind(), err.to_string()))
    }

    pub fn write_field_elements<'a>(
        &mut self,
        fes: impl IntoIterator<Item = &'a F>,
    ) -> Result<(), LibraryError>
    where
        F: 'a,
    {
        for fe in fes.into_iter() {
            self.write_field_element(fe)?;
        }
        Ok(())
    }

    fn read_field_element(&mut self) -> Result<F, LibraryError> {
        let mut repr = <F as PrimeField>::Repr::default();
        self.stream
            .read_exact(repr.as_mut())
            .map_err(|err| LibraryError::Transcript(err.kind(), err.to_string()))?;
        repr.as_mut().reverse();
        let fe = F::from_repr_vartime(repr).ok_or_else(|| {
            LibraryError::Transcript(
                std::io::ErrorKind::Other,
                "Invalid field element encoding in proof".to_string(),
            )
        })?;
        self.common_field_element(&fe)?;
        Ok(fe)
    }

    pub fn read_field_elements(&mut self, n: usize) -> Result<Vec<F>, LibraryError> {
        (0..n).map(|_| self.read_field_element()).collect()
    }
}
