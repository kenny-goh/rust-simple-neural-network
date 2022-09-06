use crate::rust_learn::activation::Activation;
use crate::rust_learn::types::Optimizer;
use crate::rust_learn::parameters::TrainParameters;
use crate::rust_learn::tensor2d::{WeightInitStrategy, Tensor2D};

/// This class needs to be refactored, its a mess.
/// The learning rate logic should be taken out of this class.
pub struct DenseLayer {
    activation: Activation,
    weights: Tensor2D,
    biases: Tensor2D,
    weights_moving_average: Option<Tensor2D>,
    biases_moving_average: Option<Tensor2D>,
}

impl DenseLayer {
    pub fn new(input_size: usize,
               num_of_nodes: usize,
               activation: Activation,
               strategy: &WeightInitStrategy) -> DenseLayer {
        let weights = Tensor2D::ndarray_random_init(num_of_nodes, input_size, &strategy);
        let biases = Tensor2D::ndarray_init_zeroes(num_of_nodes);

        return DenseLayer {
            activation,
            weights,
            biases,
            weights_moving_average: None,
            biases_moving_average: None,
        };
    }

    pub fn forward_props(&self, input: &Tensor2D) -> (Tensor2D, Tensor2D) {
        let z = &self.weights.dot(input) + &self.biases;
        let a = self.activation.compute(&z);
        (a, z)
    }

    pub fn back_props(&mut self,
                      epoch: i32,
                      train_size: f32,
                      partial_error: &Tensor2D,
                      z: Tensor2D,
                      prev_activation: &Tensor2D,
                      params: &TrainParameters,
    ) -> Tensor2D {

        // println!("Partial Error {:?}", partial_error);

        let batch_size = prev_activation.size(1);

        let (error,
            derivative_weights,
            derivative_biases) = self.calc_gradient(params, partial_error, z, &prev_activation, batch_size);

        let grad_clipping = params.gradient_clipping.as_ref();
        let next_error = if grad_clipping.is_some() {
            let (clip_min, clip_max) = grad_clipping.unwrap();
            (&self.weights.t().dot(&error)).clip(*clip_min, *clip_max)
        } else {
            (&self.weights.t()).dot(&error)
        };

        // update weights
        match params.optimizer {
            Optimizer::SGD => {
                self.update_weights_sgd(epoch, train_size, params, derivative_weights, derivative_biases);
            }
            Optimizer::SGDMomentum => {
                self.update_weights_sgd_momentum(epoch, train_size, params, derivative_weights, derivative_biases);
            }
            Optimizer::RMSProps => {
                self.update_weights_rms(epoch, train_size, params, derivative_weights, derivative_biases);
            }
        }

        // println!("Weight {}", &self.weights);

        return next_error;
    }

    fn update_weights_rms(&mut self, epoch: i32, train_size: f32, params: &TrainParameters, derivative_weights: Tensor2D, derivative_biases: Tensor2D) {

        if self.weights_moving_average.is_none() {
            self.weights_moving_average = Some(&self.weights * 0.)
        }
        if self.biases_moving_average.is_none() {
            self.biases_moving_average = Some(&self.biases * 0.)
        }

        let learning_rate = (1. / (1. + params.learning_rate_decay * epoch as f32)) * params.learning_rate;
        let sdw = self.weights_moving_average.as_ref().unwrap() * params.momentum + &derivative_weights.powi(2) * (1. - params.momentum);
        let sdb = self.biases_moving_average.as_ref().unwrap() * params.momentum + &derivative_biases.powi(2) * (1. - params.momentum);

        self.weights = if params.l2 > 0. {
            // regularization
            (&self.weights * (1. - (learning_rate * params.l2 / train_size)) ) - (&derivative_weights / &sdw.sqrt() * learning_rate)
        } else {
            &self.weights - &(&derivative_weights / &sdw.sqrt() * learning_rate)
        };

        self.biases = &self.biases - &(&derivative_biases / &sdb.sqrt() * learning_rate);

        self.weights_moving_average = Some(sdw.clone());
        self.biases_moving_average = Some(sdb.clone());
    }

    fn update_weights_sgd_momentum(&mut self, epoch: i32,
                                   train_size: f32,
                                   params: &TrainParameters,
                                   derivative_weights: Tensor2D,
                                   derivative_biases: Tensor2D) {
        if self.weights_moving_average.is_none() {
            self.weights_moving_average = Some(&self.weights * 0.)
        }
        if self.biases_moving_average.is_none() {
            self.biases_moving_average = Some(&self.biases * 0.)
        }

        let vdw = self.weights_moving_average.as_ref().unwrap() * params.momentum + &derivative_weights * (1. - params.momentum);
        let vdb = self.biases_moving_average.as_ref().unwrap() * params.momentum + &derivative_biases * (1. - params.momentum);

        let learning_rate = (1. / (1. + params.learning_rate_decay * epoch as f32)) * params.learning_rate;

        self.weights = if params.l2 > 0. {
            (&self.weights * (1. - (learning_rate * params.l2 / train_size)) ) - (&vdw * learning_rate)
        } else {
            &self.weights - &(&vdw * learning_rate)
        };

        self.biases = &self.biases - &(&vdb * learning_rate);

        self.weights_moving_average = Some(vdw.clone());
        self.biases_moving_average = Some(vdb.clone());
    }

    fn update_weights_sgd(&mut self, epoch: i32, train_size: f32, params: &TrainParameters, derivative_weights: Tensor2D, derivative_biases: Tensor2D) {
        let learning_rate = (1. / (1. + params.learning_rate_decay * epoch as f32)) * params.learning_rate;
        self.weights = if params.l2 > 0. {
            (&self.weights * (1. - (learning_rate * params.l2 / train_size)) ) - (learning_rate * derivative_weights)
        } else {
            &self.weights - &(learning_rate * derivative_weights)
        };
        self.biases = &self.biases - &(learning_rate * derivative_biases);
    }

    fn calc_gradient(&mut self,
                     params: &TrainParameters,
                     partial_error: &Tensor2D,
                     z: Tensor2D,
                     prev_activation: &Tensor2D,
                     batch_size: f32) -> (Tensor2D, Tensor2D, Tensor2D) {
        let grad_clipping = params.gradient_clipping.as_ref();
        return if grad_clipping.is_some() {
            let (clip_min, clip_max) = grad_clipping.unwrap();
            let error = self.activation.compute_derivative(z) * partial_error.clip(*clip_min, *clip_max);
            let derivative_weights = ((error.dot(&(prev_activation.t()))) * batch_size).clip(*clip_min, *clip_max);
            let derivative_biases = (error.sum_keep_dims() / batch_size).clip(*clip_min, *clip_max);
            (error, derivative_weights, derivative_biases)
        } else {
            let error = self.activation.compute_derivative(z) * partial_error;
            let derivative_weights = error.dot(&(prev_activation.t())) / batch_size;
            let derivative_biases = error.sum_keep_dims() / batch_size;
            (error, derivative_weights, derivative_biases)
        };
    }

    pub fn get_weights(&self) -> &Tensor2D {
        &self.weights
    }
    pub fn get_biases(&self) -> &Tensor2D {
        &self.biases
    }
    pub fn set_weights(&mut self, weights: &Tensor2D) {
        self.weights = weights.clone();
    }
    pub fn set_biases(&mut self, biases: &Tensor2D) {
        self.biases = biases.clone();
    }
}
