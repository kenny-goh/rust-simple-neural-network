use std::{f32, fs};
use std::path::Path;
use ndarray::{arr2, array, Array2};
use crate::algorithm::nn_layer::{Activation, CostType, NeuralNet};
use crate::utils::Utils;

const SCALING_DURATION: f32 = 180.;
const SCALING_CLICK: f32 = 100.;
const SCALING_ERROR: f32 = 100.;
const SCALING_KEYPRESS: f32 = 200.;
const SCALING_MOUSE_MOTION: f32 = 3000.;
const SCALING_TYPING_SPEED: f32 = 2000.;

pub fn login_bot_detector() {

    let layer_dims = vec![6usize, 30usize, 1usize];
    let layer_activation = vec![Activation::Tanh,
                                Activation::Tanh,
                                Activation::Sigmoid];

    let raw_file_content =
        fs::read_to_string("./data/login_data.csv").expect("File is missing!");

    let dataset: Vec<&str> = raw_file_content.lines().collect();

    let (x_predict, y_predict) = split_test_data(&dataset, 0.1);


    let parameters;
    if Path::new("./model/login_bot.json").exists() {
        println!("Model exists, loading model from file.");
        parameters = Utils::deserialize("./model/login_bot.json").expect("Unable to deserialize xor json");
    }
    else {
        println!("Model model does not exits, training from scratch...");

        let (x_train, y_train) = split_training_data(&dataset, 0.9);

        parameters = NeuralNet::train(&x_train,
                                      &y_train,
                                      layer_dims,
                                      &layer_activation,
                                      1.2, 10000,
                                      &CostType::CrossEntropy,
                                      true);
        // Utils::serialize(&parameters, "./model/login_bot.json").unwrap();
    }

    let predictions = NeuralNet::predict(&parameters, &layer_activation, &x_predict);
    let Y = y_predict.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });

    println!("Test Accuracy: {} %", NeuralNet::calc_accuracy(&Y, &predictions));

    let x_single:Array2<f32> = array![[0. / SCALING_DURATION,
        2. / SCALING_CLICK,
        180. / SCALING_KEYPRESS,
        50.  / SCALING_MOUSE_MOTION,
        0. / SCALING_TYPING_SPEED,
        0. / SCALING_ERROR]].reversed_axes();
    let result = NeuralNet::predict(&parameters, &layer_activation,&x_single);
    println!("{:?}", result);
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
        let row_x: [f32; 6] = [
            f32::min(chunks.get(0).unwrap().parse::<f32>().unwrap() / SCALING_DURATION, 1.),
            f32::min(chunks.get(1).unwrap().parse::<f32>().unwrap() / SCALING_CLICK, 1.),
            f32::min(chunks.get(2).unwrap().parse::<f32>().unwrap() / SCALING_KEYPRESS, 1.),
            f32::min(chunks.get(3).unwrap().parse::<f32>().unwrap() / SCALING_MOUSE_MOTION, 1.),
            f32::min(chunks.get(4).unwrap().parse::<f32>().unwrap() / SCALING_TYPING_SPEED, 1.),
            f32::min(chunks.get(5).unwrap().parse::<f32>().unwrap() / SCALING_ERROR, 1.)
        ];
        let row_y: [f32; 1] = [chunks.get(6).unwrap().parse::<f32>().unwrap()];
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
        let row_x: [f32; 6] = [
            f32::min(chunks.get(0).unwrap().parse::<f32>().unwrap() / SCALING_DURATION, 1.),
            f32::min(chunks.get(1).unwrap().parse::<f32>().unwrap() / SCALING_CLICK, 1.),
            f32::min(chunks.get(2).unwrap().parse::<f32>().unwrap() / SCALING_KEYPRESS, 1.),
            f32::min(chunks.get(3).unwrap().parse::<f32>().unwrap() / SCALING_MOUSE_MOTION, 1.),
            f32::min(chunks.get(4).unwrap().parse::<f32>().unwrap() / SCALING_TYPING_SPEED, 1.),
            f32::min(chunks.get(5).unwrap().parse::<f32>().unwrap() / SCALING_ERROR, 1.)
        ];
        let row_y: [f32; 1] = [chunks.get(6).unwrap().parse::<f32>().unwrap()];
        rows_x.push(row_x);
        rows_y.push(row_y);
    }

    let X = arr2(rows_x[..].try_into().unwrap()).reversed_axes();
    let Y = arr2(rows_y[..].try_into().unwrap()).reversed_axes();
    (X, Y)
}
