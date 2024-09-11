use crate::FieldBinding;
use cudarc::driver::DeviceRepr;
use ff::{Field, PrimeField};
use halo2curves::{bn256::Fr, serde::SerdeObject};
use itertools::Itertools;

pub trait FieldBindingConversion<F> {
    type FieldBinding: Copy + DeviceRepr + Send + Sync + Unpin;

    fn from_canonical_form(b: Self::FieldBinding) -> F;

    fn from_montgomery_form(b: Self::FieldBinding) -> F;

    fn to_canonical_form(f: F) -> Self::FieldBinding;

    fn to_montgomery_form(f: F) -> Self::FieldBinding;
}

macro_rules! field_binding_conversion_impl {
    ($field:ident, $field_binding:ident) => {
        impl FieldBindingConversion<$field> for $field {
            type FieldBinding = $field_binding;

            fn from_canonical_form(b: FieldBinding) -> $field {
                $field::from_raw(b.data)
            }

            fn from_montgomery_form(b: FieldBinding) -> $field {
                let bytes = b
                    .data
                    .into_iter()
                    .map(|data| data.to_le_bytes())
                    .collect_vec()
                    .concat();
                let value = $field::from_raw_bytes_unchecked(bytes.as_ref());
                // This multiplication of identity is for doing montgomery reduction
                // if value is larger than modulus, it will subtract modulus
                value * $field::ONE
            }

            fn to_canonical_form(f: $field) -> FieldBinding {
                let repr = f.to_repr();
                let bytes = repr.as_ref();
                let data = bytes
                    .chunks(8)
                    .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                    .collect_vec();
                FieldBinding {
                    data: data.try_into().unwrap(),
                }
            }

            fn to_montgomery_form(f: $field) -> FieldBinding {
                let mut buf = vec![];
                f.write_raw(&mut buf).unwrap();
                let data = buf
                    .chunks(8)
                    .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                    .collect_vec();
                FieldBinding {
                    data: data.try_into().unwrap(),
                }
            }
        }
    };
}

#[cfg(feature = "bn254")]
field_binding_conversion_impl!(Fr, FieldBinding);
