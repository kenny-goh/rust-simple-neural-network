#[allow(dead_code, unused_imports)]

use std::path::Path;
use ndarray::{array, Array2};
use crate::deep_learning::depecated::nn::{Activation, CostType, NeuralNetDeprecated};
use crate::utils::Utils;

/// XOR
/// [0, 0] = [0]
/// [0, 1] = [1]
/// [1, 0] = [1]
/// [1, 1] - [0]
///
pub fn xor_example() {

    let layer_dims = vec![2usize, 40usize,1usize];
    let layer_activation = vec![Activation::LeakyRelu,
                                Activation::LeakyRelu,
                                Activation::Sigmoid];

    let x_predict:Array2<f32> = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.],[0., 1.],[0., 1.],[0., 1.]].reversed_axes();
    let y_predict:Array2<f32> = array![[0.], [1.], [1.], [0.],[1.],[1.],[1.]].reversed_axes();

    let parameters;
    if Path::new("./model/xor.json").exists() {
        println!("XOR model exists, loading model from file.");
        parameters = Utils::deserialize("./model/xor.json").expect("Unable to deserialize xor json");
    }
    else {
        println!("XOR model does not exits, training from scratch...");
        let x_train:Array2<f32> = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.],[1., 1.],[1., 1.],[1., 1.]].reversed_axes();
        let y_train:Array2<f32> = array![[0.], [1.], [1.], [0.], [0.], [0.], [0.]].reversed_axes();

        parameters = NeuralNetDeprecated::train(&x_train,
                                                &y_train,
                                                layer_dims,
                                                &layer_activation,
                                                1.2,
                                                1000000,
                                                &CostType::CrossEntropy,
                                                true);

        // Utils::serialize(&parameters, "./model/xor.json").unwrap();
    }

    let predictions = NeuralNetDeprecated::predict(&parameters, &layer_activation, &x_predict);
    let y = y_predict.mapv(|a| if a > 0.5 { 1.0 } else { 0.0 });

    println!("predict: {:?}", predictions);
    println!("Accuracy: {} %", NeuralNetDeprecated::calc_accuracy(&y, &predictions));

    let x_single:Array2<f32> = array![[1., 1.]].reversed_axes();
    let result = NeuralNetDeprecated::predict_as_probability(&parameters, &layer_activation, &x_single);
    println!("{:?}", result);
}


