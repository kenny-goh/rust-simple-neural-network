use ndarray::Array2;
use crate::deep_learning::activation::Activation;
use crate::deep_learning::dense_layer::DenseLayer;
use serde::{Serialize, Deserialize};

pub enum MetaLayer {
    Dense(usize, Activation),
}

pub enum Layer {
    Dense(DenseLayer),
}