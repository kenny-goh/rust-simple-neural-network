#[allow(dead_code, unused_imports)]

use rust_deep_learning::rust_learn;
use crate::rust_learn::activation::Activation;
use crate::rust_learn::costs::Cost;
use crate::rust_learn::neural_net::{NeuralNet};
use crate::rust_learn::optimizer::Optimizer;
use crate::rust_learn::parameters::TrainParameters;
use crate::rust_learn::tensor2d::{WeightInitStrategy, Tensor2D};
use crate::rust_learn::types::MetaLayer;
use std::path::Path;
use ndarray::{array, Array2};
use colored::*;

/// XOR
/// [0, 0] = [0]
/// [0, 1] = [1]
/// [1, 0] = [1]
/// [1, 1] - [0]
#[cfg(test)]
pub fn xor_example() {

    let mut net = NeuralNet::new(2,&[
        MetaLayer::Dense(10, Activation::LeakyRelu),
        MetaLayer::Dense(1, Activation::Sigmoid)],
                                  &WeightInitStrategy::Xavier
    );

    let x_predict = Tensor2D::NDArray(array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]].reversed_axes());
    let y_predict = Tensor2D::NDArray(array![[0.], [1.], [1.], [0.]].reversed_axes());

    if Path::new("./model/xor.json").exists() {
        println!("XOR model exists, loading model from file.");
        net.load_weights("./model/xor.json");
    }
    else {
        println!("XOR model does not exits, training from scratch...");
        let x_train = Tensor2D::NDArray(array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]].reversed_axes());
        let y_train = Tensor2D::NDArray(array![[0.], [1.], [1.], [0.], [0.]].reversed_axes());

        net.train(&x_train,
                   &y_train,
                   &TrainParameters::default()
                       .cost(Cost::CrossEntropy)
                       .learning_rate(0.05)
                       .batch_size(1)
                       .optimizer_rms_props(0.9)
                       .batch_size(4)
                       .iterations(Some(2000))
                       .target_stop_condition(Some(99.99999))
        );
        net.save_weights("./model/xor.json");
    }

    let predictions = net.predict(&x_predict);
    let y = y_predict.to_binary_value(0.5);
    println!("Test Accuracy: {} %", NeuralNet::calculate_accuracy(&y, &predictions).to_string().bold());

    let p1 = net.predict(&Tensor2D::NDArray(array![[1.,1.]].reversed_axes()));
    println!("Result for [0,0]: {}", p1);
    assert_eq!(p1, Tensor2D::NDArray(array![[0.]]));

    let p2 = net.predict(&Tensor2D::NDArray(array![[0.,1.]].reversed_axes()));
    println!("Result for [0,1] {}", p2);
    assert_eq!(p1, Tensor2D::NDArray(array![[1.]]));

    let p3 = net.predict(&Tensor2D::NDArray(array![[1.,0.]].reversed_axes()));
    println!("Result for [1,0] {}", p3);
    assert_eq!(p1, Tensor2D::NDArray(array![[1.]]));

    let p4 = net.predict(&Tensor2D::NDArray(array![[0.,0.]].reversed_axes()));
    println!("Result for [0,0] {}", p4);
    assert_eq!(p1, Tensor2D::NDArray(array![[0.]]));

}


