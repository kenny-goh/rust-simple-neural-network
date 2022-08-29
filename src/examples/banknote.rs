use std::fs;
use std::path::Path;
use ndarray::{arr2, array, Array2, Axis};
use crate::algorithm::ndarray_helper;
use crate::algorithm::nn_layer::{Activation, CostType};
use crate::utils::Utils;
use crate::algorithm::nn_layer::NeuralNet;

pub fn bank_note_auth_example() {

    let layer_dims = vec![4usize, 30usize, 1usize];
    let layer_activation = vec![Activation::LeakyRelu,
                                Activation::LeakyRelu,
                                Activation::Sigmoid];

    let raw_file_content =
        fs::read_to_string("./data/example.txt").expect("File is missing!");

    let dataset: Vec<&str> = raw_file_content.lines().collect();

    let (x_predict, y_predict) = split_test_data(&dataset, 0.1);

    let parameters;
    if Path::new("./model/banknote_deep.json").exists() {
        println!("Bank note model exists, loading model from file.");
        parameters = Utils::deserialize("./model/banknote_deep.json").expect("Unable to deserialize xor json");
    }
    else {
        println!("Bank note model does not exits, training from scratch...");
        let (x_train, y_train) = split_training_data(&dataset, 0.9);

        parameters = NeuralNet::train(&x_train,
                                      &y_train,
                                      layer_dims,
                                      &layer_activation,
                                      1.2,
                                      10000,
                                      &CostType::CrossEntropy,
                                      true);

        // Utils::serialize(&parameters, "./model/banknote_deep.json").unwrap();
    }

    let predictions = NeuralNet::predict(&parameters, &layer_activation,&x_predict);
    let Y = y_predict.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });

    println!("Test Accuracy: {} %", NeuralNet::calc_accuracy(&Y, &predictions));

    // let x_single:Array2<f32> = array![[-1.8411,10.8306,2.769,-3.0901]].reversed_axes();
    // let result = NeuralNet::predict_as_probability(&parameters, &x_single);
    // println!("{:?}", result);
}

///
fn split_training_data(lines: &Vec<&str>, split_ratio: f32) -> (Array2<f32>, Array2<f32>) {
    let size = lines.len();
    let split_index = (size as f32 * split_ratio) as usize;
    let lhs = &lines[..split_index];
    let mut rows_x = vec![];
    let mut rows_y = vec![];
    for str in lhs {
        let chunks: Vec<&str> = str.split(",").collect();
        let col1 = chunks.get(0).unwrap().parse::<f32>().unwrap();
        let col2 = chunks.get(1).unwrap().parse::<f32>().unwrap();
        let col3 = chunks.get(2).unwrap().parse::<f32>().unwrap();
        let col4 = chunks.get(3).unwrap().parse::<f32>().unwrap();
        let row_x: [f32; 4] = [
            col1, col2, col3, col4,
        ];
        let row_y: [f32; 1] = [chunks.get(4).unwrap().parse::<f32>().unwrap()];
        rows_x.push(row_x);
        rows_y.push(row_y);
    }
    let X = arr2(rows_x[..].try_into().unwrap()).reversed_axes();
    let Y = arr2(rows_y[..].try_into().unwrap()).reversed_axes();
    (X, Y)
}

///
fn split_test_data(lines: &Vec<&str>, split_ratio: f32) -> (Array2<f32>, Array2<f32>) {
    let size = lines.len();
    let split_index = (size as f32 * split_ratio) as usize;
    let rhs = &lines[split_index..lines.len()];
    let mut rows_x = vec![];
    let mut rows_y = vec![];
    for str in rhs {
        let chunks: Vec<&str> = str.split(",").collect();
        let col1 = chunks.get(0).unwrap().parse::<f32>().unwrap();
        let col2 = chunks.get(1).unwrap().parse::<f32>().unwrap();
        let col3 = chunks.get(2).unwrap().parse::<f32>().unwrap();
        let col4 = chunks.get(3).unwrap().parse::<f32>().unwrap();
        let row_x: [f32; 4] = [
            col1, col2, col3, col4,
        ];
        let row_y: [f32; 1] = [chunks.get(4).unwrap().parse::<f32>().unwrap()];
        rows_x.push(row_x);
        rows_y.push(row_y);
    }

    let X = arr2(rows_x[..].try_into().unwrap()).reversed_axes();
    let Y = arr2(rows_y[..].try_into().unwrap()).reversed_axes();
    (X, Y)
}