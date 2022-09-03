use std::collections::HashMap;
use std::fs::File;
use ndarray::{arr2, Array, Array2, ArrayBase, Axis, Dim, Ix, Ix2, OwnedRepr};
use ndarray_rand::rand_distr::num_traits::real::Real;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use itertools::izip;
use colored::*;
use crate::deep_learning::activation::Activation;
use crate::deep_learning::costs::Cost;
use crate::deep_learning::dense_layer::DenseLayer;
use crate::deep_learning::tensor2d::{RandomWeightInitStrategy, Tensor2D};
use crate::deep_learning::types::{Layer, MetaLayer};
use std::io::{Write, Error, Read};
use crate::deep_learning::optimizer::{Optimizer};
use crate::deep_learning::parameters::TrainParameters;

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
                 parameter: &TrainParameters) {

        let mut previous_costs: f32 = 0.;
        let mut no_improvement_count: usize = 0;
        let train_size = *(&x_input.size(1));
        let batches = NeuralNet::split_to_batches(&x_input, &y_output, parameter.batch_size);
        let mut batch_index = 0;
        let mut iterations_index = 0;
        let mut halt = false;
        let mut accuracy_dev = 0_f32;
        let mut epoch = 0;
        let mut costs = 0_f32;
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

                let mut partial_error = parameter.cost.derivative(batch_output, activation_last);
                for (layer, (a, z)) in izip!(l_iter, output_iter) {
                    partial_error = Self::layer_back_props(epoch, train_size, &mut partial_error, layer, &a, z, parameter);
                }

                costs = parameter.cost.cost(&activation_last, &batch_output);
                // use a percentage
                if costs == previous_costs {
                    no_improvement_count += 1;
                } else {
                    no_improvement_count = 0;
                }

                previous_costs = costs;
                batch_index += 1;
                iterations_index += 1;

                if no_improvement_count >= parameter.stop_no_improvement {
                    println!("\n{}\n","********* Halt due to no improvement ************".red());
                    println!("{:?}", activation_last);
                    halt = true;
                    break;
                }

                let predicted = self.predict(&x_input);
                accuracy_dev = NeuralNet::calculate_accuracy(&predicted, y_output);
                println!("Batch {:5} Accuracy: {:5.4} Costs: {:.8}", batch_index.to_string().green(), accuracy_dev.to_string().green(), costs.to_string().green());

                if parameter.target_stop_condition.is_some() {
                    if accuracy_dev >= *parameter.target_stop_condition.as_ref().unwrap() {
                        println!("\n{}\n", "********* Halt due target stop condition met ************".blue());
                        halt = true;
                        break;
                    }
                }

                if parameter.iterations.is_some() {
                    if iterations_index >  *parameter.iterations.as_ref().unwrap() {
                        println!("\n{}\n", "********* Halt due iterations exceeded ************".blue());
                        halt = true;
                        break;
                    }
                }
            }
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

    fn layer_back_props(epoch: i32, train_size: f32, partial_error: &mut Tensor2D, layer: &mut Layer, a: &Tensor2D, z: Option<Tensor2D>, params: &TrainParameters) -> Tensor2D {
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

    fn split_to_batches(
        input: &Tensor2D,
        output: &Tensor2D,
        batch_size: usize,
    ) -> Vec<(Tensor2D, Tensor2D)> {
        let m = input.size(1);
        let batch_nums = (m as f32 / batch_size as f32).ceil() as usize;
        let mut batches: Vec<(Tensor2D, Tensor2D)> = Vec::with_capacity(batch_nums);
        for i in 0..batch_nums - 1 {
            let batch_from = (i * batch_size);
            let batch_to = (batch_from + batch_size+1);
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