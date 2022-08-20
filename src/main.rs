use ndarray::prelude::*;
use std::fs;

mod lib;

use lib::*;

fn bank_note_auth_example() {
    let raw_file_content =
        fs::read_to_string("./example.txt").expect("File is missing!");

    let dataset: Vec<&str> = raw_file_content.lines().collect();

    let (x_train, y_train) = NeuralNetwork3Layer::split_training_data(&dataset, 0.9);

    // println!("X TRAIN SHAPE {:?}", &x_train.shape());
    // println!("Y TRAIN SHAPE {:?}", &y_train.shape());

    let (x_predict, y_predict) = NeuralNetwork3Layer::split_test_data(&dataset, 0.1);

    let parameters = NeuralNetwork3Layer::train(&x_train, &y_train, 20, 50000, true);
    let predictions = NeuralNetwork3Layer::predict(&parameters, &x_predict);
    let Y = y_predict.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });

    println!("Accuracy: {} %", NeuralNetwork3Layer::calc_accuracy(&Y, &predictions));
}

/// XOR
/// [0, 0] = [0]
/// [0, 1] = [1]
/// [1, 0] = [1]
/// [1, 1] - [0]
fn xor_example() {
    let x_train:Array2<f64> = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.],[1., 1.],[1., 1.],[1., 1.]].reversed_axes();
    let y_train:Array2<f64> = array![[0.], [1.], [1.], [0.], [0.], [0.], [0.]].reversed_axes();

    let x_predict:Array2<f64> = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.],[0., 1.],[0., 1.],[0., 1.]].reversed_axes();
    let y_predict:Array2<f64> = array![[0.], [1.], [1.], [0.],[1.],[1.],[1.]].reversed_axes();

    let parameters = NeuralNetwork3Layer::train(&x_train, &y_train, 10, 2000, true);
    let predictions = NeuralNetwork3Layer::predict(&parameters, &x_predict);
    let Y = y_predict.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });

    println!("predict: {:?}", predictions);
    println!("Accuracy: {} %", NeuralNetwork3Layer::calc_accuracy(&Y, &predictions));
}


fn main() {
    // bank_note_auth_example();
    xor_example();
}


