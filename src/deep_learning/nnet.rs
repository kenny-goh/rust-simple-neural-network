use std::collections::HashMap;
use std::fs::File;
use ndarray::{arr2, Array, Array2, ArrayBase, Axis, Dim, Ix, Ix2, OwnedRepr};
use ndarray_rand::rand_distr::num_traits::real::Real;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use itertools::izip;
use crate::deep_learning::activation::Activation;
use crate::deep_learning::costs::Cost;
use crate::deep_learning::dense_layer::DenseLayer;
use crate::deep_learning::tensor2d::{RandomWeightInitStrategy, Tensor2D};
use crate::deep_learning::types::{Layer, MetaLayer};
use std::io::{Write, Error, Read};

pub struct NeuralNet {
    input_size: usize,
    layers: Vec<Layer>,
}

impl NeuralNet {
    pub fn new(input_size: usize,
               metalayers: &[MetaLayer],
               weight_init_strategy: &RandomWeightInitStrategy) -> NeuralNet {
        let mut layer_inputs = input_size;
        let mut layers: Vec<Layer> = Vec::with_capacity(metalayers.len());
        let mut metalayers_iter = metalayers.iter();

        let first_layer = metalayers_iter.next().unwrap();
        match first_layer {
            MetaLayer::Dense(size, activation) => {
                layers.push(Layer::Dense(DenseLayer::new(layer_inputs, *size, *activation, weight_init_strategy)));
                layer_inputs = *size;
            }
        }

        for meta_layer in metalayers_iter {
            match meta_layer {
                MetaLayer::Dense(size, activation) => {
                    layers.push(Layer::Dense(DenseLayer::new(layer_inputs, *size, *activation, weight_init_strategy)));
                    layer_inputs = *size;
                }
            }
        }

        NeuralNet {
            input_size,
            layers,
        }
    }

    pub fn train(&mut self, x_input: &Tensor2D,
                 y_output: &Tensor2D,
                 parameter: &TrainingParameter) {

        let mut previous_costs: f32 = 0.;
        let mut no_improvement_count: usize = 0;

        // train loop
        for i in 0..parameter.iterations {
            // forward props
            let mut output_layers: Vec<(Tensor2D, Option<Tensor2D>)> = Vec::with_capacity(self.layers.len());
            let mut input: Tensor2D = x_input.clone();

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

            let mut partial_error = parameter.cost.derivative(y_output, activation_last);
            for (layer, (a, z)) in izip!(l_iter, output_iter) {
                partial_error = Self::layer_back_props(parameter.learning_rate, &mut partial_error, layer, &a, z);
            }
            let costs = parameter.cost.cost(&activation_last, &y_output);
            if costs == previous_costs {
                no_improvement_count += 1;
            } else {
                no_improvement_count = 0;
            }
            previous_costs = costs;

            if i % parameter.log_interval == 0 {
                let al_binary = activation_last.to_binary_value(0.5);
                let y_output_binary = y_output.to_binary_value(0.5);
                let accuracy = NeuralNet::calculate_accuracy(&y_output_binary, &al_binary);
                println!("Training [iteration: {:7}] Cost: {:.8} Train Accuracy: {:.2}", i, costs, accuracy);
                if accuracy >= 100.0 {
                    break;
                }
            }

            if no_improvement_count >= parameter.stop_no_improvement_iterations {
                println!("Stopping training due to abort if no improvement for {} iterations condition",
                         parameter.stop_no_improvement_iterations);
                break
            }
        }
    }

    fn layer_back_props(learning_rate: f32, partial_error: &mut Tensor2D, layer: &mut Layer, a: &Tensor2D, z: Option<Tensor2D>) -> Tensor2D {
        match layer {
            Layer::Dense(dense_layer) => dense_layer.back_props(
                partial_error,
                z.unwrap(),
                &a,
                learning_rate,
            ),
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

    pub fn calculate_accuracy(target: &Tensor2D, actual: &Tensor2D) -> f32 {
        // pretty sure its possible to do this via matrix operations.
        let size = target.size(1);
        let mut index: usize = 0;
        let mut matches = 0_f32;
        let vec2 = actual.to_raw_vec();
        for x in target.to_raw_vec() {
            if x == *vec2.get(index).unwrap() {
                matches += 1.;
            }
            index += 1;
        }
        (matches / size) * 100.0
    }


    // fixme: implement me
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

    // fixme: implement me
    pub fn save_model(path: &str) {}

    // fixme: implement me
    pub fn load_model(path: &str) {}

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

pub struct TrainingParameter {
    pub learning_rate: f32,
    pub iterations: usize,
    pub cost: Cost,
    pub log_interval: usize,
    pub stop_training_target_accuracy_treshold: f32,
    pub stop_no_improvement_iterations: usize,
    pub save_snashot: bool,
    pub save_snapshot_interval: usize,
}

impl TrainingParameter {
    pub fn default() -> TrainingParameter {
        TrainingParameter {
            learning_rate: 1.0,
            log_interval: 100,
            cost: Cost::CrossEntropy,
            iterations: 10000,
            stop_training_target_accuracy_treshold: 100.0,
            stop_no_improvement_iterations: 50,
            save_snashot: false,
            save_snapshot_interval: 0,
        }
    }
    pub fn learning_rate(&mut self, learning_rate: f32) -> &mut TrainingParameter {
        self.learning_rate = learning_rate;
        self
    }

    pub fn stop_training_target_accuracy_treshold(&mut self, target_treshold: f32) -> &mut TrainingParameter {
        self.stop_training_target_accuracy_treshold = target_treshold;
        self
    }

    pub fn iterations(&mut self, iterations: usize) -> &mut TrainingParameter {
        self.iterations = iterations;
        self
    }
    pub fn stop_no_improvement_iterations(&mut self, iterations: usize) -> &mut TrainingParameter {
        self.stop_no_improvement_iterations = iterations;
        self
    }

    pub fn cost(&mut self, cost: Cost) -> &mut TrainingParameter {
        self.cost = cost;
        self
    }

}


