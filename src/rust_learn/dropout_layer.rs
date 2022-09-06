use ndarray::{arr2, Array2};
use crate::rust_learn::parameters::TrainParameters;
use crate::rust_learn::tensor2d::{Tensor2D};

/// A dropout layer.
pub struct DropoutLayer {
    pub probability: f32,
    mask: Tensor2D,
}
impl DropoutLayer {

    pub fn new(probability: f32) -> DropoutLayer {
        DropoutLayer {
            probability,
            mask: Tensor2D::NDArray(Array2::ones((1,1)))
        }
    }

    pub fn forward_props(&mut self, input: &Tensor2D, training: bool) -> Tensor2D {
        let rows = input.shape()[0];
        let cols = input.shape()[1];
        let prob = if training { self.probability } else { 0. };
        self.mask = Tensor2D::ndarray_random_mask(rows, cols, prob);
        input * &self.mask
    }

    pub fn back_props(&mut self,
                      partial_error: &Tensor2D) -> Tensor2D {
        partial_error * &self.mask
    }
}
