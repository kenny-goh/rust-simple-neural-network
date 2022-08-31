use ndarray::{arr2, Array, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::deep_learning::activation::Activation;
use crate::deep_learning::tensor2d::{RandomWeightInitStrategy, Tensor2D};

pub struct DenseLayer {
    activation: Activation,
    biases: Tensor2D,
    weights: Tensor2D,
}


impl DenseLayer {
    pub fn new(input_size: usize, num_of_nodes: usize, activation: Activation, strategy: &RandomWeightInitStrategy) -> DenseLayer {
        let weights= Tensor2D::init_random_as_ndarray(num_of_nodes, input_size, &strategy);
        let biases = Tensor2D::init_zeros_as_ndarray(num_of_nodes);

        return DenseLayer {
            activation,
            weights,
            biases,
        };
    }

    pub fn forward_props(&self, a: &Tensor2D) -> (Tensor2D, Tensor2D) {
        let z = &self.weights.dot(a) + &self.biases;
        let a = self.activation.compute(&z);
        (a, z)
    }

    pub fn back_props(&mut self,
                      partial_error: &Tensor2D,
                      z: Tensor2D,
                      prev_activation: &Tensor2D,
                      learning_rate: f32,
    ) -> Tensor2D {
        let training_set_length = prev_activation.size(1);

        let error = self.activation.compute_derivative(z) * partial_error;

        let derivative_weights = (error.dot(&prev_activation.t())) * (1.0 / training_set_length);
        let derivative_biases = Tensor2D::NDArray(arr2(&[[error.sum()]])) * (1.0 / training_set_length);

        self.weights = &self.weights - &(derivative_weights * learning_rate);
        self.biases = &self.biases - &(derivative_biases * learning_rate);

        let next_error = (&self.weights.t()).dot(&error);

        return next_error;
    }

    pub fn get_weights(&self)->&Tensor2D {
        &self.weights
    }

    pub fn get_biases(&self)->&Tensor2D {
        &self.biases
    }

    pub fn set_weights(&mut self, weights: &Tensor2D) {
        self.weights = weights.clone();
    }

    pub fn set_biases(&mut self, biases: &Tensor2D) {
        self.biases = biases.clone();
    }
}
