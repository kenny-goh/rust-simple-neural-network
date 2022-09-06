use crate::rust_learn::activation::Activation;
use crate::rust_learn::dense_layer::DenseLayer;
use crate::rust_learn::dropout_layer::DropoutLayer;

pub enum MetaLayer {
    Dense(usize, Activation),
    Dropout(f32),
}

pub enum Layer {
    Dense(DenseLayer),
    Dropout(DropoutLayer),
}

pub enum Optimizer {
    SGD,
    SGDMomentum,
    RMSProps
}
