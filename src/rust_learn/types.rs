use crate::rust_learn::activation::Activation;
use crate::rust_learn::dense_layer::DenseLayer;

pub enum MetaLayer {
    Dense(usize, Activation),
}

pub enum Layer {
    Dense(DenseLayer),
}