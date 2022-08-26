use ndarray::Array2;
use ndarray::Axis;
use ndarray::{arr2, Array};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::collections::HashMap;
use std::ops::Neg;

pub struct NeuralNet {}

impl NeuralNet {
    /// Check layer sizes
    /// Arguments:
    ///  X : input dataset of shape (input size, number of examples)
    ///  Y : labels of shape (output size, number of examples)
    ///
    /// Returns:
    ///  n_x : the size of the input layer
    ///  n_y : the size of the output layer
    pub fn layer_sizes(X: &Array2<f64>, Y: &Array2<f64>) -> (usize, usize) {
        let n_x = X.len_of(Axis(0));
        let n_y = Y.len_of(Axis(0));
        (n_x, n_y)
    }

    /// Initialize parameters
    /// Argument:
    ///  n_x : size of the input layer
    ///  n_h : size of the hidden layer
    ///  n_y : size of the output layer
    ///
    /// Returns:
    ///  params :  map containing parameters:
    ///  W1 : weight matrix of shape (n_h, n_x)
    ///  b1 : bias vector of shape (n_h, 1)
    ///  W2 : weight matrix of shape (n_y, n_h)
    ///  b2 : bias vector of shape (n_y, 1)
    pub fn init_parameters(n_x: usize, n_h: usize, n_y: usize) -> HashMap<String, Array2<f64>> {
        let mut parameters = HashMap::new();
        let W1 = Array::random((n_h, n_x), Uniform::new(0., 1.)) * 0.001_f64;
        let W2 = Array::random((n_y, n_h), Uniform::new(0., 1.)) * 0.001_f64;
        parameters.insert("W1".to_string(), W1);
        parameters.insert("W2".to_string(), W2);
        return parameters;
    }

    /**
     * Perform forward propagation
     * Argument:
     *  X : input data of size (n_x, m)
     *  parameters : map containing  parameters (output of initialization function)
     *
     * Returns:
     *  A2 : The sigmoid output of the second activation
     *  cache : a map containing "Z1", "A1", "Z2" and "A2"
     */
    pub fn forward_propagation(
        X: &Array2<f64>,
        parameters: &HashMap<String, Array2<f64>>,
    ) -> (Array2<f64>, HashMap<String, Array2<f64>>) {
        let W1 = parameters.get("W1").expect("Expected W1 but not found");
        let W2 = parameters.get("W2").expect("Expected W2 but not found");

        let Z1 = W1.dot(X);
        let A1 = NeuralNet::tanh(&Z1);
        let Z2 = W2.dot(&A1);
        let A2 = NeuralNet::sigmoid(&Z2);

        let mut cache = HashMap::new();
        cache.insert("Z1".to_string(), Z1);
        cache.insert("A1".to_string(), A1);
        cache.insert("Z2".to_string(), Z2);
        cache.insert("A2".to_string(), A2.clone());

        (A2, cache)
    }

    fn sigmoid(Z2: &Array2<f64>) -> Array2<f64> {
        1. / (1. + (Z2.mapv(|x| (-x).exp())))
    }

    fn tanh(Z1: &Array2<f64>) -> Array2<f64> {
        let e = Z1.mapv(|x| x.exp());
        let neg_e = Z1.mapv(|x| x.neg().exp());
        (&e - &neg_e) / (&e + &neg_e)
    }

    /// Implement the backward propagation
    ///
    /// Arguments:
    ///  parameters :  map containing our parameters
    ///  cache : a map containing "Z1", "A1", "Z2" and "A2".
    ///  X : input data of shape (2, number of examples)
    ///  Y : "true" labels vector of shape (1, number of examples)
    ///
    /// Returns:
    ///  grads : map containing gradients with respect to different parameters
    ///
    pub fn backward_propagation(
        parameters: &HashMap<String, Array2<f64>>,
        cache: &HashMap<String, Array2<f64>>,
        X: &Array2<f64>,
        Y: &Array2<f64>,
    ) -> HashMap<String, Array2<f64>> {
        let m = X.len_of(Axis(1)) as f64;
        let W1 = parameters.get("W1").expect("Expected W2 but not found");
        let W2 = parameters.get("W2").expect("Expected W2 but not found");
        let A1 = cache.get("A1").expect("Expected A1 but not found");
        let A2 = cache.get("A2").expect("Expected A2 but not found");

        //  Backward propagation: calculate dW1, db1, dW2, db2.
        let dZ2 = A2 - Y;
        let dW2 = (&dZ2.dot(&A1.t())) * (1.0 / m);
        let dZ1 = (W2.t().dot(&dZ2)) * (1.0 - (A1.mapv(|x| x.powf(2.0))));
        let dW1 = (&dZ1.dot(&X.t())) * (1.0 / m);

        let mut result = HashMap::new();
        result.insert("dW1".to_string(), dW1);
        result.insert("dW2".to_string(), dW2);
        result
    }

    /**
     * Calculate loss between Y and A2
     *
     * Arguments:
     *  A2 : The sigmoid output of the second activation, of shape (1, number of examples)
     *  Y : "true" labels vector of shape (1, number of examples)
     *
     * Returns:
     *  cost : simple loss function
     */
    pub fn compute_cost(A2: &Array2<f64>, Y: &Array2<f64>) -> f64 {
        (A2 - Y).sum().abs()
    }

    /// Updates parameters using the gradient descent update rule given above
    ///
    /// Arguments:
    ///  parameters : map containing  parameters
    ///  grads : map containing  gradients
    /// Returns:
    ///  parameters : map containing updated parameters
    pub fn update_parameters(
        parameters: &HashMap<String, Array2<f64>>,
        grads: &HashMap<String, Array2<f64>>,
        learning_rate: f64,
    ) -> HashMap<String, Array2<f64>> {
        let W1 = parameters.get("W1").expect("Expected W1 but not found");
        let W2 = parameters.get("W2").expect("Expected W2 but not found");

        let dW1 = grads.get("dW1").expect("Expected dW1 but not found");
        let dW2 = grads.get("dW2").expect("Expected dW2 but not found");

        // Update rule for each parameter
        let _W1 = W1 - dW1;
        let _W2 = W2 - dW2;

        let mut result = HashMap::new();
        result.insert("W1".to_string(), _W1);
        result.insert("W2".to_string(), _W2);
        result
    }

    /// Arguments:
    ///  X : dataset of shape (2, number of examples)
    ///  Y : labels of shape (1, number of examples)
    ///  n_h : size of the hidden layer
    ///  num_iterations : Number of iterations in gradient descent loop
    ///  print_cost : if True, print the cost every 1000 iterations
    ///
    ///  Returns:
    /// parameters : parameters learnt by the model
    pub fn train(
        X: &Array2<f64>,
        Y: &Array2<f64>,
        n_h: usize,
        num_iterations: i32,
        print_cost: bool,
    ) -> HashMap<String, Array2<f64>> {
        let (n_x, n_y) = NeuralNet::layer_sizes(&X, &Y);
        let mut parameters = NeuralNet::init_parameters(n_x, n_h, n_y);

        // Loop (gradient descent)
        for i in 0..num_iterations {
            // Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            let (A2, cache) = NeuralNet::forward_propagation(X, &parameters);

            // Cost function . Inputs : "A2, Y, parameters". Outputs: "cost".
            let cost = NeuralNet::compute_cost(&A2, &Y);

            // Backpropagation.Inputs: "parameters, cache, X, Y". Outputs: "grads".
            let grads = NeuralNet::backward_propagation(&parameters, &cache, X, Y);

            // Gradient descent parameter update . Inputs : "parameters, grads". Outputs: "parameters".
            parameters = NeuralNet::update_parameters(&parameters, &grads, 1.2);

            // Print the cost every 1000 iterations
            if print_cost && i % 100 == 0 {
                println!("Cost after iteration {}: {}", i, cost);
            }
        }
        parameters
    }

    /// Using the learned parameters, predicts a class for each example.txt in X
    ///
    /// Arguments:
    /// parameters : map containing parameters
    /// X : input data of size (n_x, m)
    ///
    /// Returns
    /// predictions : vector of predictions of the model
    ///
    pub fn predict(parameters: &HashMap<String, Array2<f64>>, X: &Array2<f64>) -> Array2<f64> {
        // Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        let (A2, cache) = NeuralNet::forward_propagation(X, parameters);
        A2.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 })
    }

    pub fn predict_as_probability(parameters: &HashMap<String, Array2<f64>>, X: &Array2<f64>) -> Array2<f64> {
        // Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        let (A2, cache) = NeuralNet::forward_propagation(X, parameters);
        A2
    }

    /// Using the learned parameters, predicts a class for each example.txt in X
    ///
    /// Arguments:
    /// parameters : map containing parameters
    /// X : input data of size (n_x, m)
    ///
    /// Returns
    /// predictions : vector of predictions of the model
    pub fn calc_accuracy(Y1: &Array2<f64>, Y2: &Array2<f64>) -> f64 {
        // pretty sure its possible to do this via matrix operations.
        let size = Y1.len_of(Axis(1)) as f64;
        let mut index: usize = 0;
        let mut matches = 0_f64;
        let vec2 = Y2.clone().into_raw_vec();
        for x in Y1.clone().into_raw_vec() {
            if x == *vec2.get(index).unwrap() {
                matches += 1.;
            }
            index += 1;
        }
        (matches / size) * 100.0
    }

    ///
    pub fn split_training_data(lines: &Vec<&str>, split_ratio: f32) -> (Array2<f64>, Array2<f64>) {
        let size = lines.len();
        let split_index = (size as f32 * split_ratio) as usize;
        let lhs = &lines[..split_index];
        let mut rows_x = vec![];
        let mut rows_y = vec![];
        for str in lhs {
            let chunks: Vec<&str> = str.split(",").collect();
            let row_x: [f64; 4] = [
                chunks.get(0).unwrap().parse::<f64>().unwrap(),
                chunks.get(1).unwrap().parse::<f64>().unwrap(),
                chunks.get(2).unwrap().parse::<f64>().unwrap(),
                chunks.get(3).unwrap().parse::<f64>().unwrap(),
            ];
            let row_y: [f64; 1] = [chunks.get(4).unwrap().parse::<f64>().unwrap()];
            rows_x.push(row_x);
            rows_y.push(row_y);
        }

        let X = arr2(rows_x[..].try_into().unwrap()).reversed_axes();
        let Y = arr2(rows_y[..].try_into().unwrap()).reversed_axes();
        (X, Y)
    }

    ///
    pub fn split_test_data(lines: &Vec<&str>, split_ratio: f32) -> (Array2<f64>, Array2<f64>) {
        let size = lines.len();
        let split_index = (size as f32 * split_ratio) as usize;
        let rhs = &lines[split_index..lines.len()];
        let mut rows_x = vec![];
        let mut rows_y = vec![];
        for str in rhs {
            let chunks: Vec<&str> = str.split(",").collect();
            let row_x: [f64; 4] = [
                chunks.get(0).unwrap().parse::<f64>().unwrap(),
                chunks.get(1).unwrap().parse::<f64>().unwrap(),
                chunks.get(2).unwrap().parse::<f64>().unwrap(),
                chunks.get(3).unwrap().parse::<f64>().unwrap(),
            ];
            let row_y: [f64; 1] = [chunks.get(4).unwrap().parse::<f64>().unwrap()];
            rows_x.push(row_x);
            rows_y.push(row_y);
        }

        let X = arr2(rows_x[..].try_into().unwrap()).reversed_axes();
        let Y = arr2(rows_y[..].try_into().unwrap()).reversed_axes();
        (X, Y)
    }
}
