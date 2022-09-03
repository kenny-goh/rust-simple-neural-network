#[allow(dead_code, unused_imports)]
use std::{f32, fs};
use std::path::Path;
use ndarray::{arr2, array, Array2};
use colored::*;
use crate::deep_learning::activation::Activation;
use crate::deep_learning::costs::Cost;
use crate::deep_learning::neural_net::NeuralNet;
use crate::deep_learning::parameters::TrainParameters;
use crate::deep_learning::tensor2d::{RandomWeightInitStrategy, Tensor2D};
use crate::deep_learning::types::MetaLayer;
use crate::utils::Utils;

const SCALING_DURATION: f32 = 180.;
const SCALING_CLICK: f32 = 100.;
const SCALING_ERROR: f32 = 100.;
const SCALING_KEYPRESS: f32 = 200.;
const SCALING_MOUSE_MOTION: f32 = 3000.;
const SCALING_TYPING_SPEED: f32 = 2000.;

pub fn login_bot_detector() {


    let mut net = NeuralNet::new(6, &[
        MetaLayer::Dense(30, Activation::LeakRelu),
        MetaLayer::Dense(1, Activation::Sigmoid)],
                                 &RandomWeightInitStrategy::Xavier
    );
    let raw_file_content =
        fs::read_to_string("./data/login_data.csv").expect("File is missing!");

    let dataset: Vec<&str> = raw_file_content.lines().collect();

    let (x_predict, y_predict) = split_test_data(&dataset, 0.1);


    if Path::new("./model/login_bot.json").exists() {
        println!("Model exists, loading model from file.");
        net.load_weights("./model/login_bot.json");
    }
    else {
        println!("Model model does not exits, training from scratch...");

        let (x_train, y_train) = split_training_data(&dataset, 0.9);

        net.train(&x_train,
                  &y_train,
                  &TrainParameters::default()
                      .cost(Cost::CrossEntropy)
                      .learning_rate(0.05)
                      .learning_rate_decay(0.5)
                      .l2(0.01)
                      .optimizer_rms_props(0.9)
                      .batch_size(32)
                      .iterations(Some(1000))
                      .target_stop_condition(None)
                  //.gradient_clipping(Some((-1.,1.)))
        );
        net.save_weights("./model/login_bot.json");
    }

    let predictions = net.predict(&x_predict);
    let y = y_predict.to_binary_value(0.5);
    println!("Test Accuracy: {} %", NeuralNet::calculate_accuracy(&y, &predictions).to_string().bold());
    let x_single:Array2<f32> = array![[100. / SCALING_DURATION,
        3. / SCALING_CLICK,
        70. / SCALING_KEYPRESS,
        3.  / SCALING_MOUSE_MOTION,
        700. / SCALING_TYPING_SPEED,
        3. / SCALING_ERROR]].reversed_axes();
    let result = net.predict_as_prob(&Tensor2D::NDArray(x_single));
    println!("{:?}", result.to_binary_value(0.5));
}

///
fn split_training_data(lines: &Vec<&str>, split_ratio: f32) -> (Tensor2D, Tensor2D){
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

    let x = arr2(rows_x[..].try_into().unwrap()).reversed_axes();
    let y = arr2(rows_y[..].try_into().unwrap()).reversed_axes();
    (Tensor2D::NDArray(x), Tensor2D::NDArray(y))
}
