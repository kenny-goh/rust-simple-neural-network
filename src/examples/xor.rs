use std::fs;
use std::path::Path;
use ndarray::{array, Array2};
use crate::{Utils};
use crate::algorithm::three_layer::NeuralNet;
use crate::utils::Utils;

/// XOR
/// [0, 0] = [0]
/// [0, 1] = [1]
/// [1, 0] = [1]
/// [1, 1] - [0]
pub fn xor_example() {

    let x_predict:Array2<f64> = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.],[0., 1.],[0., 1.],[0., 1.]].reversed_axes();
    let y_predict:Array2<f64> = array![[0.], [1.], [1.], [0.],[1.],[1.],[1.]].reversed_axes();

    let parameters;
    if Path::new("./model/xor.json").exists() {
        println!("XOR model exists, loading model from file.");
        parameters = Utils::deserialize("./model/xor.json").expect("Unable to deserialize xor json");
    }
    else {
        println!("XOR model does not exits, training from scratch...");
        let x_train:Array2<f64> = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.],[1., 1.],[1., 1.],[1., 1.]].reversed_axes();
        let y_train:Array2<f64> = array![[0.], [1.], [1.], [0.], [0.], [0.], [0.]].reversed_axes();

        parameters = NeuralNet::train(&x_train, &y_train, 10, 2000, true);

        Utils::serialize(&parameters, "./model/xor.json").unwrap();
    }

    let predictions = NeuralNet::predict(&parameters, &x_predict);
    let Y = y_predict.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });

    println!("predict: {:?}", predictions);
    println!("Accuracy: {} %", NeuralNet::calc_accuracy(&Y, &predictions));

    let x_single:Array2<f64> = array![[1., 1.]].reversed_axes();
    let result = NeuralNet::predict_as_probability(&parameters, &x_single);
    println!("{:?}", result);
}


