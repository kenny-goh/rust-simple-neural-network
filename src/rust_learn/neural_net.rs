use std::collections::HashMap;
use std::fs::File;
use itertools::izip;
use colored::*;
use crate::rust_learn::dense_layer::DenseLayer;
use crate::rust_learn::tensor2d::{WeightInitStrategy, Tensor2D};
use crate::rust_learn::types::{Layer, MetaLayer};
use std::io::{Write, Error, Read};
use std::time::Instant;
use ndarray::s;
use crate::rust_learn::parameters::TrainParameters;

/// NeuralNet is a configurable deep learning 'model' that can run training on input data and labels
/// to make a prediction.
///
/// Note: This object is in a state of flux and will be be unstable until version 1.00
pub struct NeuralNet {
    input_size: usize,
    layers: Vec<Layer>,
}

impl NeuralNet {

    /// Instantiate a new NeuralNet model.
    ///
    /// # Arguments
    /// * `input_size` - Size of feature input
    /// * `metalayers` - Array of NN layer
    /// * `weight_init_strategy` - enum of initialization strategy e.g Random, Xavier
    /// # Examples
    /// ```
    /// use rust_deep_learning::rust_learn::activation::Activation;
    /// use rust_deep_learning::rust_learn::neural_net::NeuralNet;
    /// use rust_deep_learning::rust_learn::tensor2d::WeightInitStrategy;
    /// use rust_deep_learning::rust_learn::types::MetaLayer;
    ///
    /// let mut net = NeuralNet::new(2,
    ///   &[MetaLayer::Dense(10, Activation::LeakyRelu),
    ///     MetaLayer::Dense(1,  Activation::Sigmoid)],
    ///     &WeightInitStrategy::Xavier);
    /// ```
    ///
    pub fn new(input_size: usize,
               metalayers: &[MetaLayer],
               weight_init_strategy: &WeightInitStrategy) -> NeuralNet {
        let mut layer_inputs = input_size;
        let mut layers: Vec<Layer> = Vec::with_capacity(metalayers.len());
        let mut metalayers_iter = metalayers.iter();

        let first_layer = metalayers_iter.next().unwrap();
        match first_layer {
            MetaLayer::Dense(size, activation) => {
                layers.push(Layer::Dense(DenseLayer::new(layer_inputs,
                                                         *size,
                                                         *activation,
                                                         weight_init_strategy)));
                layer_inputs = *size;
            }
        }

        for meta_layer in metalayers_iter {
            match meta_layer {
                MetaLayer::Dense(size, activation) => {
                    layers.push(Layer::Dense(DenseLayer::new(layer_inputs,
                                                             *size,
                                                             *activation,
                                                             weight_init_strategy)));
                    layer_inputs = *size;
                }
            }
        }

        NeuralNet {
            input_size,
            layers,
        }
    }

    /// Runs training on the given input feature and label with the given training parameters
    /// # Arguments
    /// * `x_input` -  2 dimensional matrix with rows of features
    /// * `y_output` - 2 dimensional matrix with rows of labels
    /// * `params` - Training parameters
    ///
    /// # Examples
    /// ```
    /// use ndarray::array;
    /// use rust_deep_learning::rust_learn::parameters::TrainParameters;
    /// use rust_deep_learning::rust_learn::tensor2d::Tensor2D;
    ///
    /// let x_train = Tensor2D::NDArray(array![
    ///   [0., 0.],
    ///   [0., 1.],
    ///   [1., 0.],
    ///   [1., 1.]
    ///  ].reversed_axes());
    ///
    ///  let y_train = Tensor2D::NDArray(array![
    ///   [0.],
    ///   [1.],
    ///   [1.],
    ///   [0.],
    ///   [0.]
    /// ].reversed_axes());
    ///
    /// net.train(&x_train,&y_train, &TrainParameters::default());
    ///
    /// ```
    pub fn train(&mut self,
                 x_input: &Tensor2D,
                 y_output: &Tensor2D,
                 params: &TrainParameters) {

        let mut previous_costs: f32 = 0.;
        let mut no_improvement_count: usize = 0;
        let train_size = *(&x_input.size(1));
        let batches = NeuralNet::split_to_batches(&x_input, &y_output, params.batch_size);
        let mut batch_index = 0;
        let mut iterations_index = 0;
        let mut halt = false;
        let mut accuracy_dev = 0_f32;
        let mut epoch = 0;
        let mut costs = 0_f32;
        let now = Instant::now();
        let eval_set_name = if params.evaluation_dataset.is_some() { "Test"} else { "Dev" };

        loop {
            println!("{}","*****************************************");
            println!(" Epoch: {}", epoch.to_string().bold());
            println!("{}", "*****************************************");
            for (batch_input, batch_output) in &batches {

                // forward props
                let mut output_layers: Vec<(Tensor2D, Option<Tensor2D>)> = Vec::with_capacity(self.layers.len());
                let mut input: Tensor2D = batch_input.clone();

                for layer in self.layers.iter_mut() {
                    let (a, z) = Self::layer_forward_props(&mut input, layer);
                    output_layers.push((input, z));
                    input = a;
                }
                output_layers.push((input, None));

                // backward pops
                let mut output_iter = output_layers.into_iter().rev();
                let l_iter = self.layers.iter_mut().rev();
                let activation_last = &output_iter.next().unwrap().0;

                // println!("ACTIVATION LAST {:?}", activation_last);

                let mut partial_error = params.cost.derivative(batch_output, activation_last);

                // println!("PARTIAL ERROR LAST {:?}", partial_error);

                for (layer, (a, z)) in izip!(l_iter, output_iter) {
                    // println!("PREV A {:?}", &a);
                    // println!("PREV Z {:?}", &z.as_ref().unwrap());
                    partial_error = Self::layer_back_props(epoch, train_size, &mut partial_error, layer, &a, z, params);
                }

                costs = params.cost.cost(&activation_last, &batch_output);
                // use a percentage
                if costs == previous_costs {
                    no_improvement_count += 1;
                } else {
                    no_improvement_count = 0;
                }

                previous_costs = costs;
                batch_index += 1;
                iterations_index += 1;

                // if no_improvement_count >= params.stop_no_improvement {
                //     println!("\n{}\n","********* Halt due to no improvement ************".red());
                //     halt = true;
                //     break;
                // }

                if iterations_index % params.log_interval == 0 {
                    if params.evaluation_dataset.is_some() {
                        let (eval_input, eval_output ) = params.evaluation_dataset.as_ref().unwrap();
                        let predicted = self.predict(&eval_input);
                        accuracy_dev = NeuralNet::calculate_accuracy(eval_output, &predicted);
                    } else {
                        let predicted = self.predict(&batch_input);
                        accuracy_dev = NeuralNet::calculate_accuracy(batch_output, &predicted);
                    }
                    println!("Batch {:5} {} Accuracy: {:5.4} Costs: {:.8}", batch_index.to_string().green(), eval_set_name, accuracy_dev.to_string().green(), costs.to_string().green());
                }

                if params.target_stop_condition.is_some() {
                    if accuracy_dev >= *params.target_stop_condition.as_ref().unwrap() {
                        println!("\n{}\n", "********* Halt due target stop condition met ************".blue());
                        halt = true;
                        break;
                    }
                }

                if params.iterations.is_some() {
                    if iterations_index >  *params.iterations.as_ref().unwrap() {
                        println!("\n{}\n", "********* Halt due iterations exceeded ************".blue());
                        halt = true;
                        break;
                    }
                }
            }

            println!("Total time in millis: {}", now.elapsed().as_millis());
            println!("Number of epoch: {}", epoch.to_string().bold());
            println!("Number of iterations: {}", iterations_index.to_string().bold());
            println!("Dev Accuracy: {} %", accuracy_dev.to_string().bold());

            if halt {
                break
            }
            epoch += 1;
            batch_index = 0;
        }
    }

    /// Predict an output given a feature
    /// # Arguments
    /// * `x_input` -  2 dimensional matrix with rows of features
    ///
    /// # Output
    /// * `Tensor2D` - Predictions represented as rows of labels
    ///
    /// # Examples
    /// ```
    /// use std::intrinsics::likely;
    /// use ndarray::array;
    /// use rust_deep_learning::rust_learn::tensor2d::Tensor2D;
    ///
    /// // Lets use a XOR example
    /// let x_test = Tensor2D::NDArray(array![
    ///   [0., 0.],
    ///   [0., 1.],
    ///   [1., 0.],
    ///   [1., 1.]
    /// ].reversed_axes());
    ///
    /// let result = net.predict(x_test);
    /// // Result will look something like
    /// // [[0],[1],[1],[0]]
    ///
    /// ```
    pub fn predict(&mut self, x_input: &Tensor2D) -> Tensor2D {
        let mut output_layers: Vec<(Tensor2D, Option<Tensor2D>)> = Vec::with_capacity(self.layers.len());
        let mut activation = x_input.clone();
        for layer in self.layers.iter_mut() {
            let (a, z) = match layer {
                Layer::Dense(dense_layer) => {
                    let (a, z) = dense_layer.forward_props(&activation);
                    (a, Some(z))
                }
            };
            output_layers.push((activation, z));
            activation = a;
        }
        activation.to_binary_value(0.5)
    }

    fn layer_back_props(epoch: i32,
                        train_size: f32,
                        partial_error: &mut Tensor2D,
                        layer: &mut Layer,
                        a: &Tensor2D,
                        z: Option<Tensor2D>,
                        params: &TrainParameters) -> Tensor2D {
        match layer {
            Layer::Dense(dense_layer) => {
                dense_layer.back_props(
                    epoch,
                    train_size,
                    partial_error,
                    z.unwrap(),
                    &a,
                    params,
                )
            }
        }
    }

    fn layer_forward_props(input: &mut Tensor2D, layer: &mut Layer) -> (Tensor2D, Option<Tensor2D>) {
        let (a, z) = match layer {
            Layer::Dense(dense_layer) => {
                let (a, z) = dense_layer.forward_props(input);
                (a, Some(z))
            }
        };
        (a, z)
    }

    pub fn predict_as_prob(&mut self, x_input: &Tensor2D) -> Tensor2D {
        let mut output_layers: Vec<(Tensor2D, Option<Tensor2D>)> = Vec::with_capacity(self.layers.len());
        let mut activation = x_input.clone();
        for layer in self.layers.iter_mut() {
            let (a, z) = match layer {
                Layer::Dense(dense_layer) => {
                    let (a, z) = dense_layer.forward_props(&activation);
                    (a, Some(z))
                }
            };
            output_layers.push((activation, z));
            activation = a;
        }
        activation
    }

    pub fn calculate_accuracy(target: &Tensor2D, actual: &Tensor2D) -> f32 {
        let matches = target.count_equal_rows(&actual);
        let size = target.size(1);
        return (matches / size) * 100.0;
    }

    // Save the model as weights. Not this method will not preserve the parameters and
    // and architecture.
    pub fn save_weights(&mut self, filename: &str) {
        let mut params: HashMap<String, Tensor2D> = HashMap::new();
        let mut layer_index = 1;
        for layer in self.layers.iter_mut() {
            match layer {
                Layer::Dense(dense_layer) => {
                    let w = dense_layer.get_weights();
                    let b = dense_layer.get_biases();
                    params.insert(format!("w{}", layer_index), w.clone());
                    params.insert(format!("b{}", layer_index), b.clone());
                    layer_index += 1;
                }
            };
        }
        NeuralNet::serialize(&params, filename).unwrap();
    }

    // Load the model using weights
    pub fn load_weights(&mut self, filename: &str) {
        let params = NeuralNet::deserialize(filename).expect("Unable to load weights");
        let mut layer_index = 1;
        for layer in self.layers.iter_mut() {
            match layer {
                Layer::Dense(dense_layer) => {
                    let w = params.get(&format!("w{}", layer_index)).unwrap();
                    let b = params.get(&format!("b{}", layer_index)).unwrap();
                    dense_layer.set_weights(w);
                    dense_layer.set_biases(b);
                    layer_index += 1;
                }
            };
        }
        NeuralNet::serialize(&params, filename).unwrap();
    }

    /// Helper method to split training data to smaller batches
    fn split_to_batches(
        input: &Tensor2D,
        output: &Tensor2D,
        batch_size: usize,
    ) -> Vec<(Tensor2D, Tensor2D)> {
        let m = input.size(1);
        let batch_nums = (m as f32 / batch_size as f32).ceil() as usize;
        let mut batches: Vec<(Tensor2D, Tensor2D)> = Vec::with_capacity(batch_nums);
        for i in 0..batch_nums - 1 {
            let batch_from = i * batch_size;
            let batch_to = batch_from + batch_size+1;
            let input_batch: Tensor2D = input.cols(batch_from as i32, batch_to as i32);
            let output_batch: Tensor2D = output.cols(batch_from as i32, batch_to as i32);
            batches.push((input_batch, output_batch));
        }
        let batch_from = (batch_nums - 1) * batch_size;
        let in_batch = input.cols(batch_from as i32, (m) as i32);
        let out_batch = output.cols(batch_from as i32, (m) as i32);
        batches.push((in_batch, out_batch));
        return batches;
    }

    // // fixme: implement me
    // pub fn save_model(path: &str) {}
    //
    // // fixme: implement me
    // pub fn load_model(path: &str) {}

    pub fn serialize(params: &HashMap<String, Tensor2D>, filename: &str) ->Result<(), Error> {
        let serialized_json = serde_json::to_string(&params).unwrap();
        let path =  filename;
        let mut output = File::create(path)?;
        write!(output, "{}", serialized_json)?;
        Ok(())
    }

    pub fn deserialize(filename: &str)->Result<HashMap<String,Tensor2D>, Error>  {
        let mut file = File::open(filename)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let params: HashMap<String,Tensor2D> = serde_json::from_str(&contents)?;
        Ok(params)
    }

}