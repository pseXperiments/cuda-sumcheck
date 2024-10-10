use crate::{FieldBinding, QuadraticExtFieldBinding};
use ff::PrimeField;
use goldilocks::{Goldilocks, GoldilocksExt2};
use halo2curves::serde::SerdeObject;

impl From<Goldilocks> for FieldBinding {
    fn from(f: Goldilocks) -> FieldBinding {
        let repr = f.to_repr();
        let bytes: &[u8] = repr.as_ref();
        FieldBinding {
            data: u64::from_le_bytes(bytes.try_into().unwrap()),
        }
    }
}

impl From<FieldBinding> for Goldilocks {
    fn from(b: FieldBinding) -> Self {
        let bytes = b.data.to_le_bytes();
        Goldilocks::from_raw_bytes_unchecked(bytes.as_ref())
    }
}

impl From<&FieldBinding> for Goldilocks {
    fn from(b: &FieldBinding) -> Self {
        let bytes = b.data.to_le_bytes();
        Goldilocks::from_raw_bytes_unchecked(bytes.as_ref())
    }
}

impl From<GoldilocksExt2> for QuadraticExtFieldBinding {
    fn from(value: GoldilocksExt2) -> Self {
        QuadraticExtFieldBinding {
            data: [
                FieldBinding::from(value.0[0]).data,
                FieldBinding::from(value.0[1]).data,
            ],
        }
    }
}

impl From<QuadraticExtFieldBinding> for GoldilocksExt2 {
    fn from(value: QuadraticExtFieldBinding) -> Self {
        GoldilocksExt2([
            Goldilocks::from(value.data[0]),
            Goldilocks::from(value.data[1]),
        ])
    }
}

impl From<&QuadraticExtFieldBinding> for GoldilocksExt2 {
    fn from(value: &QuadraticExtFieldBinding) -> Self {
        GoldilocksExt2([
            Goldilocks::from(value.data[0]),
            Goldilocks::from(value.data[1]),
        ])
    }
}
