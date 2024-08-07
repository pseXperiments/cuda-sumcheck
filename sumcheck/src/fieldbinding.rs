use crate::FieldBinding;
use ff::{Field, PrimeField};
use halo2curves::{bn256::Fr, serde::SerdeObject};
use itertools::Itertools;

pub trait FromFieldBinding<F> {
    fn from_canonical_form(b: FieldBinding) -> F;

    fn from_montgomery_form(b: FieldBinding) -> F;
}

pub trait ToFieldBinding<F> {
    fn to_canonical_form(f: F) -> FieldBinding;

    fn to_montgomery_form(f: F) -> FieldBinding;
}

macro_rules! field_binding_conversion {
    ($field:ident) => {
        impl FromFieldBinding<$field> for $field {
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
        }

        impl ToFieldBinding<$field> for $field {
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

field_binding_conversion!(Fr);
