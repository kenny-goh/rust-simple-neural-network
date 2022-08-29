use std::collections::HashMap;
use ndarray::{arr2, Array, Array2, ArrayBase, Axis, OwnedRepr, ShapeBuilder, Ix2, s, Dim, Ix};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand_isaac::Isaac64Rng;
use ndarray_rand::rand::SeedableRng;
use crate::algorithm::ndarray_helper::NDArrayHelper;

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

pub struct NeuralNet {}

impl NeuralNet {
    fn init_parameters(layer_dims: Vec<usize>) -> HashMap<String, Matrix2D> {

        // let seed = 42;
        // let mut rng = Isaac64Rng::seed_from_u64(seed);

        let mut parameters = HashMap::new();
        let L = layer_dims.len();
        for l in 1..L {
            let W: Matrix2D = Array::random((layer_dims[l], layer_dims[l - 1]), Uniform::new(-1., 1.)) * 0.001_f32;
            parameters.insert(format!("W{}", l), W);
            parameters.insert(format!("b{}", l), Array2::zeros((layer_dims[l], 1)));
            // assert_eq!(parameters.get(&format!("W{}", l)).unwrap().shape().to_vec(), [layer_dims[l], layer_dims[l - 1]]);
        }

        parameters
    }

    fn linear_forward(A: &Matrix2D, parameters: &HashMap<String, Matrix2D>, layer_index: usize) -> LinearForwardTuple {
        let W = parameters.get(&format!("W{}", layer_index)).unwrap();
        let b = parameters.get(&format!("b{}", layer_index)).unwrap();

        let Z = W.dot(A) + b;
        let cache = (A.clone(), layer_index);
        return (Z, cache);
    }

    fn linear_activation_forward(A_prev: &Matrix2D,
                                 parameters: &HashMap<String, Matrix2D>,
                                 layer_index: usize,
                                 activation: &Activation) -> LinearActivationForwardTuple {
        return match activation {
            Activation::Sigmoid => {
                let (Z, linear_cache) = NeuralNet::linear_forward(A_prev, parameters, layer_index);
                let (A, activation_cache) = NeuralNet::sigmoid(&Z);
                let cache = (linear_cache, activation_cache.clone());
                (A, cache)
            }
            Activation::LeakyRelu => {
                let (Z, linear_cache) = NeuralNet::linear_forward(A_prev, parameters, layer_index);
                let (A, activation_cache) = NeuralNet::leaky_relu(&Z);
                let cache = (linear_cache, activation_cache.clone());
                (A, cache)
            }
            Activation::Tanh => {
                let (Z, linear_cache) = NeuralNet::linear_forward(A_prev, parameters, layer_index);
                let (A, activation_cache) = NeuralNet::tanh(&Z);
                let cache = (linear_cache, activation_cache.clone());
                (A, cache)
            }
        };
    }

    fn l_model_forward(X: &Matrix2D,
                       parameters: &HashMap<String, Matrix2D>,
                       layer_activations: &Vec<Activation>) -> LinearModelForwardTuple {
        let mut caches: Vec<LinearForwardAndActivationCache> = vec![];
        let mut A = X.clone();
        let L = parameters.len() / 2;

        //Implement[LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in 1..L {
            let A_prev = A.clone();

            let (A_, cache) = NeuralNet::linear_activation_forward(
                &A_prev,
                parameters,
                l,
                layer_activations.get(l).unwrap());
            A = A_;
            caches.push(cache);
        }

        // Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        let (AL, cache) = NeuralNet::linear_activation_forward(
            &A,
            parameters,
            L,
            layer_activations.get(L).unwrap());

        caches.push(cache);

        (AL, caches)
    }

    fn linear_backward(parameters: &HashMap<String, Matrix2D>, cache: &LinearForwardCache, dZ: &Matrix2D) -> LinearBackwardTriple {
        let (A_prev, layer_index) = cache;

        let W = parameters.get(&format!("W{}", layer_index)).unwrap();
        let b = parameters.get(&format!("b{}", layer_index)).unwrap();

        let m = A_prev.len_of(Axis(1)) as f32;

        let dW = (dZ.dot(&A_prev.t())) * (1.0 / m);
        let db = arr2(&[[dZ.sum()]]) * (1.0 / m);

        let dA_prev = W.t().dot(dZ);

        (dA_prev, dW, db)
    }

    fn linear_activation_backward(parameters: &HashMap<String, Matrix2D>,
                                  dA: &Matrix2D,
                                  cache: &LinearForwardAndActivationCache,
                                  activation: &Activation) -> LinearActivationBackwardTriple {
        let (linear_cache, activation_cache) = cache;
        return match activation {
            Activation::LeakyRelu => {
                let dZ = NeuralNet::leaky_relu_backward(&dA, &activation_cache);
                let (dA_prev, dW, db) = NeuralNet::linear_backward(parameters, linear_cache, &dZ);
                (dA_prev, dW, db)
            }
            Activation::Sigmoid => {
                let dZ = NeuralNet::sigmoid_backward(&dA, &activation_cache);
                let (dA_prev, dW, db) = NeuralNet::linear_backward(parameters, linear_cache, &dZ);
                (dA_prev, dW, db)
            }
            Activation::Tanh => {
                let dZ = NeuralNet::tanh_backward(&dA, &activation_cache);
                let (dA_prev, dW, db) = NeuralNet::linear_backward(parameters, linear_cache, &dZ);
                (dA_prev, dW, db)
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
                        AL: &Matrix2D,
                        Y: &Matrix2D,
                        caches: Vec<LinearForwardAndActivationCache>) -> HashMap<String, Matrix2D> {
        let mut grads = HashMap::new();
        let L = caches.len();

        let dAL = Self::derivative_cost_output(cost, AL, Y);

        // // Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        let current_cache = &caches.get(L - 1).unwrap();

        let (dA_prev, dWL, dbL) = NeuralNet::linear_activation_backward(parameters,
                                                                        &dAL,
                                                                        &current_cache,
                                                                        layer_activations.get(L).unwrap());
        grads.insert(format!("dA{}", (L - 1)), dA_prev);
        grads.insert(format!("dW{}", L), dWL);
        grads.insert(format!("db{}", L), dbL);

        // Loop from l = L - 2 to 0
        for l in (0..=(L - 2)).rev() {
            // lth layer :(RELU -> LINEAR) gradients.
            // Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            let current_cache = &caches[l];

            let dAl_plus_1 = grads.get(&format!("dA{}", (l + 1))).unwrap();

            let (dA_prev_temp, dW_temp, db_temp) = NeuralNet::linear_activation_backward(
                parameters,
                dAl_plus_1,
                current_cache,
                layer_activations.get(l).unwrap(),
            );
            grads.insert(format!("dA{}", l), dA_prev_temp);
            grads.insert(format!("dW{}", (l + 1)), dW_temp);
            grads.insert(format!("db{}", (l + 1)), db_temp);
        }
        grads
    }

    fn derivative_cost_output(cost: &CostType, AL: &Matrix2D, Y: &Matrix2D) -> ArrayBase<OwnedRepr<f32>, Dim<[Ix; 2]>> {
        return match cost {
            CostType::CrossEntropy => {
                let Y_vec = Y.iter().map(|f| *f).collect::<Vec<f32>>();
                let AL_vec = AL.iter().map(|f| *f).collect::<Vec<f32>>();

                // cross entropy
                let dAL = Y_vec
                    .iter()
                    .zip(AL_vec)
                    .map(|(y, a)| {
                        let v = -(y / a - (1.0 - y) / (1.0 - a));
                        if v.is_nan() { 0.0 } else { v }
                    })
                    .collect::<Vec<f32>>();

                let shape = AL.shape();
                let (row, col) = (shape[0], shape[1]);
                let dAL = Array::from_shape_vec((row, col), dAL).unwrap();
                dAL
            }
            CostType::Quadratic => {
                AL - Y
            }
        };
    }

    fn update_parameters(parameters_in: &HashMap<String, Matrix2D>,
                         grads: &HashMap<String, Matrix2D>,
                         learning_rate: f32) -> HashMap<String, Matrix2D> {
        let mut parameters = parameters_in.clone();
        let L = parameters.len() / 2;
        for l in 0..L {
            let l_next = l + 1;

            let W = parameters.get(&format!("W{}", l_next)).unwrap();
            let b = parameters.get(&format!("b{}", l_next)).unwrap();

            let dW = grads.get(&format!("dW{}", l_next)).unwrap();
            let db = grads.get(&format!("db{}", l_next)).unwrap();

            let _W = W - (dW * learning_rate);
            let _b = b - (db * learning_rate);

            parameters.insert(format!("W{}", l_next), _W);
            parameters.insert(format!("b{}", l_next), _b);
        }
        parameters
    }

    pub fn train(X: &Matrix2D,
                 Y: &Matrix2D,
                 layers_dims: Vec<usize>,
                 layers_activation: &Vec<Activation>,
                 learning_rate: f32,
                 iterations: usize,
                 cost_type: &CostType,
                 print_cost: bool) -> HashMap<String, Matrix2D> {
        let mut parameters = NeuralNet::init_parameters(layers_dims);

        // Loop (gradient descent)
        for i in 0..iterations {

            // Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            let (AL, caches) = NeuralNet::l_model_forward(&X, &parameters, layers_activation);

            // Compute cost.
            let cost = NeuralNet::compute_cost(&cost_type, &AL, &Y);

            // Backward propagation.
            let grads = NeuralNet::l_model_backward(cost_type, &parameters, layers_activation, &AL, &Y, caches);

            // println!("grads {:?}", grads);

            // Update parameters.
            parameters = NeuralNet::update_parameters(&parameters, &grads, learning_rate);

            // Print the cost every 100 training example
            if print_cost && i % 1 == 0 {
                let _AL = AL.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });
                let _Y = Y.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });
                let accuracy = NeuralNet::calc_accuracy(&_Y, &_AL);
                println!("[{}] Training Accuracy: {:.2}, Cost: {:.8}", i, NeuralNet::calc_accuracy(&_Y, &_AL), cost);
                if accuracy >= 100.0 {
                    break;
                }
            }
        }
        parameters
    }

    pub fn predict(parameters: &HashMap<String, Matrix2D>, layer_activations: &Vec<Activation>, X: &Matrix2D) -> Matrix2D {
        //Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        let (A, cache) = NeuralNet::l_model_forward(&X, &parameters, layer_activations);
        A.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 })
    }

    pub fn predict_as_probability(parameters: &HashMap<String, Matrix2D>, layer_activations: &Vec<Activation>, X: &Matrix2D) -> Matrix2D {
        // Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        let (A, cache) = NeuralNet::l_model_forward(&X, &parameters, layer_activations);
        A
    }

    pub fn compute_cost(cost: &CostType, AL: &Matrix2D, Y: &Matrix2D) -> f32 {
        return match cost {
            CostType::CrossEntropy => {
                let m = Y.len_of(Axis(1)) as f32;
                let a = AL.mapv(|x| f32::ln(x)) * Y;
                let b = (1. - Y) * (1. - AL).mapv(|x| f32::ln(x));
                let log_probs = a + b;
                let cost = log_probs.sum() * (-1.0 / m);
                if f32::is_nan(cost) { 0. } else { cost }
            }
            CostType::Quadratic => {
                let m = Y.len_of(Axis(1)) as f32;
                ((AL - Y).mapv(|x| f32::powi(x, 2))).sum() * 1. / m
            }
        };
    }

    fn sigmoid(Z: &Matrix2D) -> (Matrix2D, &Matrix2D) {
        let A = NDArrayHelper::sigmoid(Z);
        return (A, Z);
    }

    fn tanh(Z: &Matrix2D) -> (Matrix2D, &Matrix2D) {
        let A = NDArrayHelper::tanh(Z);
        return (A, Z);
    }

    fn leaky_relu(Z: &Matrix2D) -> (Matrix2D, &Matrix2D) {
        let A = Z.mapv(|x| f32::max(0.01, x));
        return (A, Z);
    }

    fn tanh_backward(dA: &Matrix2D, activation: &Matrix2D) -> Matrix2D {
        dA * (1.0 - (activation.mapv(|z| z.powi(2))))
    }

    fn sigmoid_backward(dA: &Matrix2D, activation_cache: &Matrix2D) -> Matrix2D {
        dA * Self::sigmoid_derivative(activation_cache)
    }

    fn sigmoid_derivative(activation: &Matrix2D) -> Matrix2D {
        //σ(x)(1−σ(x))
        (NDArrayHelper::sigmoid(activation) * (1. - NDArrayHelper::sigmoid(activation)))
    }

    fn leaky_relu_backward(dA: &Matrix2D, activation_cache: &Matrix2D) -> Matrix2D {
        dA * Self::leaky_relu_derivative(activation_cache)
    }

    fn leaky_relu_derivative(activation: &Matrix2D) -> Matrix2D {
        activation.mapv(|x| if x < 0. { 0.01 } else { 1. })
    }

    /// Using the learned parameters, predicts a class for each example.txt in X
    ///
    /// Arguments:
    /// parameters : map containing parameters
    /// X : input data of size (n_x, m)
    ///
    /// Returns
    /// predictions : vector of predictions of the model
    pub fn calc_accuracy(Y1: &Matrix2D, Y2: &Matrix2D) -> f32 {
        // pretty sure its possible to do this via matrix operations.
        let size = Y1.len_of(Axis(1)) as f32;
        let mut index: usize = 0;
        let mut matches = 0_f32;
        let vec2 = Y2.clone().into_raw_vec();
        for x in Y1.clone().into_raw_vec() {
            if x == *vec2.get(index).unwrap() {
                matches += 1.;
            }
            index += 1;
        }
        (matches / size) * 100.0
    }
}