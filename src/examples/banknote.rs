use std::fs;
use std::path::Path;
use ndarray::{array, Array2};
use crate::{Utils};
use crate::algorithm::three_layer::NeuralNet;
use crate::utils::Utils;

pub fn bank_note_auth_example() {

    let raw_file_content =
        fs::read_to_string("./example.txt").expect("File is missing!");

    let dataset: Vec<&str> = raw_file_content.lines().collect();

    let (x_predict, y_predict) = NeuralNet::split_test_data(&dataset, 0.1);

    let parameters;
    if Path::new("./model/banknote.json").exists() {
        println!("Bank note model exists, loading model from file.");
        parameters = Utils::deserialize("./model/banknote.json").expect("Unable to deserialize xor json");
    }
    else {
        println!("Bank note model does not exits, training from scratch...");

        let (x_train, y_train) = NeuralNet::split_training_data(&dataset, 0.9);
        // println!("X TRAIN SHAPE {:?}", &x_train.shape());
        // println!("Y TRAIN SHAPE {:?}", &y_train.shape());

        parameters = NeuralNet::train(&x_train, &y_train, 20, 50000, true);
        Utils::serialize(&parameters, "./model/banknote.json").unwrap();
    }

    let predictions = NeuralNet::predict(&parameters, &x_predict);
    let Y = y_predict.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });

    println!("Accuracy: {} %", NeuralNet::calc_accuracy(&Y, &predictions));

    let x_single:Array2<f64> = array![[-1.8411,10.8306,2.769,-3.0901]].reversed_axes();
    let result = NeuralNet::predict_as_probability(&parameters, &x_single);
    println!("{:?}", result);
}
