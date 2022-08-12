use ndarray::prelude::*;
use std::convert::TryInto;
use std::fs;

mod lib;

use lib::*;

fn main() {
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
