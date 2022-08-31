#[allow(dead_code, unused_imports)]

use std::fs;
use std::path::Path;
use ndarray::{arr2, Array2};
use crate::deep_learning::activation::Activation;
use crate::deep_learning::costs::Cost;
use crate::deep_learning::nnet::{NeuralNet, TrainingParameter};
use crate::deep_learning::tensor2d::{RandomWeightInitStrategy, Tensor2D};
use crate::deep_learning::types::MetaLayer;

pub fn bank_note_auth_example() {

    let mut nnet = NeuralNet::new(4,&[
        MetaLayer::Dense(30, Activation::LeakRelu),
        MetaLayer::Dense(1, Activation::Sigmoid)],
        &RandomWeightInitStrategy::Xavier
    );

    let raw_file_content =
        fs::read_to_string("./data/example.txt").expect("File is missing!");

    let dataset: Vec<&str> = raw_file_content.lines().collect();

    let (x_predict, y_predict) = split_test_data(&dataset, 0.1);

    if Path::new("./model/banknote_deep.json").exists() {
        println!("Bank note model exists, loading model from file.");
        nnet.load_weights("./model/banknote_deep.json");
    }
    else {
        println!("Bank note model does not exits, training from scratch...");
        let (x_train, y_train) = split_training_data(&dataset, 0.9);

        println!("x_train shape {:?} y train shape {:?}", x_train.shape(), y_train.shape());

        nnet.train(&x_train,
                   &y_train,
                   &TrainingParameter::default()
                       .cost(Cost::CrossEntropy)
                       .learning_rate(0.05)
                       .stop_no_improvement_iterations(50)
                       .stop_training_target_accuracy_treshold(100.00)
                       .iterations(10000)
        );

        nnet.save_weights("./model/banknote_deep.json");

    }

    let predictions = nnet.predict(&x_predict);
    let y = y_predict.to_binary_value(0.5);
    println!("Test Accuracy: {} %", NeuralNet::calculate_accuracy(&y, &predictions));

}

///
fn split_training_data(lines: &Vec<&str>, split_ratio: f32) -> (Tensor2D, Tensor2D) {
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
    let x = arr2(rows_x[..].try_into().unwrap()).reversed_axes();
    let y = arr2(rows_y[..].try_into().unwrap()).reversed_axes();
    (Tensor2D::NDArray(x), Tensor2D::NDArray(y))
}

///
fn split_test_data(lines: &Vec<&str>, split_ratio: f32) -> (Tensor2D, Tensor2D) {
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

    let x = arr2(rows_x[..].try_into().unwrap()).reversed_axes();
    let y = arr2(rows_y[..].try_into().unwrap()).reversed_axes();
    (Tensor2D::NDArray(x), Tensor2D::NDArray(y))
}
