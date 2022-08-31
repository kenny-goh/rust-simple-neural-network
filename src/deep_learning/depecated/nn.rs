use std::collections::HashMap;
use ndarray::{arr2, Array, Array2, ArrayBase, Axis, OwnedRepr, Dim, Ix};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
// use rand_isaac::Isaac64Rng;
// use ndarray_rand::rand::SeedableRng;

pub type Matrix2D = Array2<f32>;

pub type LinearForwardCache = (Matrix2D, usize);
pub type LinearForwardTuple = (Matrix2D, LinearForwardCache);
pub type LinearForwardAndActivationCache = (LinearForwardCache, Matrix2D);
pub type LinearActivationForwardTuple = (Matrix2D, LinearForwardAndActivationCache);
pub type LinearModelForwardTuple = (Matrix2D, Vec<LinearForwardAndActivationCache>);
pub type LinearBackwardTriple = (Matrix2D, Matrix2D, Matrix2D);
pub type LinearActivationBackwardTriple = (Matrix2D, Matrix2D, Matrix2D);

#[derive(Debug)]
pub enum Activation {
    LeakyRelu,
    Sigmoid,
    Tanh,
}

#[derive(Debug)]
pub enum CostType {
    CrossEntropy,
    Quadratic,
}

pub struct NeuralNetDeprecated {}

impl NeuralNetDeprecated {
    fn init_parameters(layer_dims: Vec<usize>) -> HashMap<String, Matrix2D> {

        // let seed = 42;
        // let mut rng = Isaac64Rng::seed_from_u64(seed);

        let mut parameters = HashMap::new();
        let num_of_layers = layer_dims.len();
        for l in 1..num_of_layers {
            let row = layer_dims[l];
            let col = layer_dims[l - 1];
            println!("r {}, c {}", row,col);
            // xavier weights https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
            let lower =  -1./f32::sqrt(col as f32);
            let upper= 1./f32::sqrt(col as f32);
            let random_weights =  Array::random((row, col), Uniform::new(0., 1.));
            let xavier_weights = lower + random_weights * (upper - lower);
            let weights: Matrix2D = xavier_weights;
            parameters.insert(format!("W{}", l), weights);
            parameters.insert(format!("b{}", l), Array2::zeros((row, 1)));
            // assert_eq!(parameters.get(&format!("W{}", l)).unwrap().shape().to_vec(), [layer_dims[l], layer_dims[l - 1]]);
        }

        parameters
    }

    fn linear_forward(activation: &Matrix2D, parameters: &HashMap<String, Matrix2D>, layer_index: usize) -> LinearForwardTuple {
        let weights = parameters.get(&format!("W{}", layer_index)).unwrap();
        let biases = parameters.get(&format!("b{}", layer_index)).unwrap();

        let z = weights.dot(activation) + biases;
        let cache = (activation.clone(), layer_index);
        return (z, cache);
    }

    fn linear_activation_forward(a_prev: &Matrix2D,
                                 parameters: &HashMap<String, Matrix2D>,
                                 layer_index: usize,
                                 activation: &Activation) -> LinearActivationForwardTuple {
        return match activation {
            Activation::Sigmoid => {
                let (z, linear_cache) = NeuralNetDeprecated::linear_forward(a_prev, parameters, layer_index);
                let (a, activation_cache) = NeuralNetDeprecated::sigmoid(&z);
                let cache = (linear_cache, activation_cache.clone());
                (a, cache)
            }
            Activation::LeakyRelu => {
                let (z, linear_cache) = NeuralNetDeprecated::linear_forward(a_prev, parameters, layer_index);
                let (a, activation_cache) = NeuralNetDeprecated::leaky_relu(&z);
                let cache = (linear_cache, activation_cache.clone());
                (a, cache)
            }
            Activation::Tanh => {
                let (z, linear_cache) = NeuralNetDeprecated::linear_forward(a_prev, parameters, layer_index);
                let (a, activation_cache) = NeuralNetDeprecated::tanh(&z);
                let cache = (linear_cache, activation_cache.clone());
                (a, cache)
            }
        };
    }

    fn l_model_forward(x_input: &Matrix2D,
                       parameters: &HashMap<String, Matrix2D>,
                       layer_activations: &Vec<Activation>) -> LinearModelForwardTuple {
        let mut caches: Vec<LinearForwardAndActivationCache> = vec![];
        let mut activation = x_input.clone();
        let num_layers = parameters.len() / 2;

        //Implement[LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in 1..num_layers {
            let prev_activation = activation.clone();

            let (activation_index_l, cache) = NeuralNetDeprecated::linear_activation_forward(
                &prev_activation,
                parameters,
                l,
                layer_activations.get(l).unwrap());
            activation = activation_index_l;
            caches.push(cache);
        }

        // Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        let (activation_last, cache) = NeuralNetDeprecated::linear_activation_forward(
            &activation,
            parameters,
            num_layers,
            layer_activations.get(num_layers).unwrap());

        caches.push(cache);

        (activation_last, caches)
    }

    fn linear_backward(parameters: &HashMap<String, Matrix2D>, cache: &LinearForwardCache, derivative_z: &Matrix2D) -> LinearBackwardTriple {
        let (prev_activation, layer_index) = cache;

        let weights = parameters.get(&format!("W{}", layer_index)).unwrap();
        //let biases = parameters.get(&format!("b{}", layer_index)).unwrap();

        let number_of_instances = prev_activation.len_of(Axis(1)) as f32;

        let derivative_weights = (derivative_z.dot(&prev_activation.t())) * (1.0 / number_of_instances);
        let derivative_biases = arr2(&[[derivative_z.sum()]]) * (1.0 / number_of_instances);

        let derivative_prev_activation = weights.t().dot(derivative_z);

        (derivative_prev_activation, derivative_weights, derivative_biases)
    }

    fn linear_activation_backward(parameters: &HashMap<String, Matrix2D>,
                                  derivative_activation: &Matrix2D,
                                  cache: &LinearForwardAndActivationCache,
                                  activation: &Activation) -> LinearActivationBackwardTriple {
        let (linear_cache, activation_cache) = cache;
        return match activation {
            Activation::LeakyRelu => {
                let dz = NeuralNetDeprecated::leaky_relu_backward(&derivative_activation, &activation_cache);
                let (da_prev, dw, db) = NeuralNetDeprecated::linear_backward(parameters, linear_cache, &dz);
                (da_prev, dw, db)
            }
            Activation::Sigmoid => {
                let dz = NeuralNetDeprecated::sigmoid_backward(&derivative_activation, &activation_cache);
                let (da_prev, dw, db) = NeuralNetDeprecated::linear_backward(parameters, linear_cache, &dz);
                (da_prev, dw, db)
            }
            Activation::Tanh => {
                let dz = NeuralNetDeprecated::tanh_backward(&derivative_activation, &activation_cache);
                let (da_prev, dw, db) = NeuralNetDeprecated::linear_backward(parameters, linear_cache, &dz);
                (da_prev, dw, db)
            }
        };
    }


    //
    // // Loop from l = L - 2 to 0
    // for (l in (L - 2) downTo 0) {
    // // lth layer :(RELU -> LINEAR) gradients.
    // // Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
    // currentCache = caches[l]
    // val L_next = (l + 1).toString()
    // val (dA_prev_temp, dW_temp, db_temp) = linearActivationBackward(
    // grads["dA$L_next"]!!,
    // currentCache,
    // activation = RELU
    // )
    // grads["dA$l"] = dA_prev_temp
    // grads["dW$L_next"] = dW_temp
    // grads["db$L_next"] = db_temp
    // }
    // return grads
    //
    fn l_model_backward(cost: &CostType,
                        parameters: &HashMap<String, Matrix2D>,
                        layer_activations: &Vec<Activation>,
                        activation_last: &Matrix2D,
                        y_output: &Matrix2D,
                        caches: Vec<LinearForwardAndActivationCache>) -> HashMap<String, Matrix2D> {
        let mut grads = HashMap::new();
        let num_layers = caches.len();

        let da_last = Self::derivative_cost_output(cost, activation_last, y_output);

        // // Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        let current_cache = &caches.get(num_layers - 1).unwrap();

        let (da_prev, dw_last, db_last) = NeuralNetDeprecated::linear_activation_backward(parameters,
                                                                                          &da_last,
                                                                                          &current_cache,
                                                                                          layer_activations.get(num_layers).unwrap());
        grads.insert(format!("dA{}", (num_layers - 1)), da_prev);
        grads.insert(format!("dW{}", num_layers), dw_last);
        grads.insert(format!("db{}", num_layers), db_last);

        // Loop from l = L - 2 to 0
        for l in (0..=(num_layers - 2)).rev() {
            // lth layer :(RELU -> LINEAR) gradients.
            // Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            let current_cache = &caches[l];

            let da_next = grads.get(&format!("dA{}", (l + 1))).unwrap();

            let (da_prev_temp, dw_temp, db_temp) = NeuralNetDeprecated::linear_activation_backward(
                parameters,
                da_next,
                current_cache,
                layer_activations.get(l).unwrap(),
            );
            grads.insert(format!("dA{}", l), da_prev_temp);
            grads.insert(format!("dW{}", (l + 1)), dw_temp);
            grads.insert(format!("db{}", (l + 1)), db_temp);
        }
        grads
    }

    fn derivative_cost_output(cost: &CostType, activation_last: &Matrix2D, y_output: &Matrix2D) -> ArrayBase<OwnedRepr<f32>, Dim<[Ix; 2]>> {
        return match cost {
            CostType::CrossEntropy => {
                let y_vec = y_output.iter().map(|f| *f).collect::<Vec<f32>>();
                let al_vec = activation_last.iter().map(|f| *f).collect::<Vec<f32>>();

                // cross entropy
                let da_last = y_vec
                    .iter()
                    .zip(al_vec)
                    .map(|(y, a)| {
                        let v = -(y / a - (1.0 - y) / (1.0 - a));
                        if v.is_nan() { 0.0 } else { v }
                    })
                    .collect::<Vec<f32>>();

                let shape = activation_last.shape();
                let (row, col) = (shape[0], shape[1]);
                Array::from_shape_vec((row, col), da_last).unwrap()
            }
            CostType::Quadratic => {
                activation_last - y_output
            }
        };
    }

    fn update_parameters(parameters_in: &HashMap<String, Matrix2D>,
                         grads: &HashMap<String, Matrix2D>,
                         learning_rate: f32) -> HashMap<String, Matrix2D> {
        let mut parameters = parameters_in.clone();
        let num_layers = parameters.len() / 2;
        for l in 0..num_layers {
            let l_next = l + 1;

            let weights = parameters.get(&format!("W{}", l_next)).unwrap();
            let biases = parameters.get(&format!("b{}", l_next)).unwrap();

            let derivative_weights = grads.get(&format!("dW{}", l_next)).unwrap();
            let derivative_biases = grads.get(&format!("db{}", l_next)).unwrap();

            let weights_delta = weights - (derivative_weights * learning_rate);
            let biases_delta = biases - (derivative_biases * learning_rate);

            parameters.insert(format!("W{}", l_next), weights_delta);
            parameters.insert(format!("b{}", l_next), biases_delta);
        }
        parameters
    }

    pub fn train(x_input: &Matrix2D,
                 y_output: &Matrix2D,
                 layers_dims: Vec<usize>,
                 layers_activation: &Vec<Activation>,
                 learning_rate: f32,
                 iterations: usize,
                 cost_type: &CostType,
                 print_cost: bool) -> HashMap<String, Matrix2D> {
        let mut parameters = NeuralNetDeprecated::init_parameters(layers_dims);

        // Loop (gradient descent)
        for i in 0..iterations {

            // Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            let (activation_last, caches) = NeuralNetDeprecated::l_model_forward(&x_input, &parameters, layers_activation);

            // Compute cost.
            let cost = NeuralNetDeprecated::compute_cost(&cost_type, &activation_last, &y_output);

            // Backward propagation.
            let grads = NeuralNetDeprecated::l_model_backward(cost_type, &parameters, layers_activation, &activation_last, &y_output, caches);

            // println!("grads {:?}", grads);

            // Update parameters.
            parameters = NeuralNetDeprecated::update_parameters(&parameters, &grads, learning_rate);

            // Print the cost every 100 training example
            if print_cost && i % 1 == 0 {
                let al_binary = activation_last.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });
                let y_output_binary = y_output.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });
                let accuracy = NeuralNetDeprecated::calc_accuracy(&y_output_binary, &al_binary);
                println!("[{}] Training Accuracy: {:.2}, Cost: {:.8}", i, NeuralNetDeprecated::calc_accuracy(&y_output_binary, &al_binary), cost);
                if accuracy >= 100.0 {
                    break;
                }
            }
        }
        parameters
    }

    pub fn predict(parameters: &HashMap<String, Matrix2D>, layer_activations: &Vec<Activation>, x_input: &Matrix2D) -> Matrix2D {
        //Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        let (activation, _) = NeuralNetDeprecated::l_model_forward(&x_input, &parameters, layer_activations);
        activation.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 })
    }

    pub fn predict_as_probability(parameters: &HashMap<String, Matrix2D>, layer_activations: &Vec<Activation>, x_input: &Matrix2D) -> Matrix2D {
        // Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        NeuralNetDeprecated::l_model_forward(&x_input, &parameters, layer_activations).0
    }

    pub fn compute_cost(cost: &CostType, activation: &Matrix2D, y_output: &Matrix2D) -> f32 {
        return match cost {
            CostType::CrossEntropy => {
                let m = y_output.len_of(Axis(1)) as f32;
                let a = activation.mapv(|x| f32::ln(x)) * y_output;
                let b = (1. - y_output) * (1. - activation).mapv(|x| f32::ln(x));
                let log_probs = a + b;
                let cost = log_probs.sum() * (-1.0 / m);
                if f32::is_nan(cost) { 0. } else { cost }
            }
            CostType::Quadratic => {
                let m = y_output.len_of(Axis(1)) as f32;
                ((activation - y_output).mapv(|x| f32::powi(x, 2))).sum() * 1. / m
            }
        };
    }

    fn sigmoid(z: &Matrix2D) -> (Matrix2D, &Matrix2D) {
        let a = Self::_sigmoid(z);
        return (a, z);
    }

    fn tanh(z: &Matrix2D) -> (Matrix2D, &Matrix2D) {
        let a = Self::_tanh(z);
        return (a, z);
    }

    fn leaky_relu(z: &Matrix2D) -> (Matrix2D, &Matrix2D) {
        let a = z.mapv(|x| f32::max(0.01, x));
        return (a, z);
    }

    fn tanh_backward(da: &Matrix2D, activation: &Matrix2D) -> Matrix2D {
        da * (1.0 - (activation.mapv(|z| z.powi(2))))
    }

    fn sigmoid_backward(da: &Matrix2D, activation_cache: &Matrix2D) -> Matrix2D {
        da * Self::sigmoid_derivative(activation_cache)
    }

    fn sigmoid_derivative(activation: &Matrix2D) -> Matrix2D {
        //σ(x)(1−σ(x))
        Self::_sigmoid(activation) * (1. - Self::_sigmoid(activation))
    }

    fn leaky_relu_backward(da: &Matrix2D, activation_cache: &Matrix2D) -> Matrix2D {
        da * Self::leaky_relu_derivative(activation_cache)
    }

    fn leaky_relu_derivative(a: &Matrix2D) -> Matrix2D {
        a.mapv(|x| if x < 0. { 0.01 } else { 1. })
    }

    fn _sigmoid(z: &Array2<f32>) -> Array2<f32> {
        1. / (1. + (z.mapv(|x| (-x).exp())))
    }

    fn _tanh(z: &Array2<f32>) -> Array2<f32> {
        let e = z.mapv(|x| x.exp());
        let neg_e = z.mapv(|x| (-x).exp());
        (&e - &neg_e) / (&e + &neg_e)
    }

    /// Using the learned parameters, predicts a class for each example.txt in X
    ///
    /// Arguments:
    /// parameters : map containing parameters
    /// X : input data of size (n_x, m)
    ///
    /// Returns
    /// predictions : vector of predictions of the model
    pub fn calc_accuracy(y1: &Matrix2D, y2: &Matrix2D) -> f32 {
        // pretty sure its possible to do this via matrix operations.
        let size = y1.len_of(Axis(1)) as f32;
        let mut index: usize = 0;
        let mut matches = 0_f32;
        let vec2 = y2.clone().into_raw_vec();
        for x in y1.clone().into_raw_vec() {
            if x == *vec2.get(index).unwrap() {
                matches += 1.;
            }
            index += 1;
        }
        (matches / size) * 100.0
    }
}