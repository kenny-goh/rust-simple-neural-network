use std::path::Path;
use colored::Colorize;
use ndarray::{arr2, Array2, Axis, s};
use crate::rust_learn::activation::Activation;
use crate::rust_learn::costs::Cost;
use crate::rust_learn::neural_net::NeuralNet;
use crate::rust_learn::parameters::TrainParameters;
use crate::rust_learn::tensor2d::{WeightInitStrategy, Tensor2D};
use crate::rust_learn::types::MetaLayer;

pub fn mnist_example() {
    let (array_x_train, array_y_train) = load_mnist_dataset(false);
    let (array_x_predict, array_y_predict) = load_mnist_dataset(true);

    let subset_x_predict = array_x_predict.slice(s![0..1000,..]).to_owned().reversed_axes();
    let subset_y_predict = array_y_predict.slice(s![0..1000,..]).to_owned().reversed_axes();

    let x_train = Tensor2D::NDArray(array_x_train.reversed_axes());
    let y_train = Tensor2D::NDArray(array_y_train.reversed_axes());
    let x_predict = Tensor2D::NDArray(array_x_predict.reversed_axes());
    let y_predict = Tensor2D::NDArray(array_y_predict.reversed_axes());

    let mut net = NeuralNet::new(784, &[
        MetaLayer::Dense(1000, Activation::Relu),
        MetaLayer::Dense(500, Activation::Relu),
        MetaLayer::Dense(10, Activation::Sigmoid)],
                     &WeightInitStrategy::Xavier
    );

    if Path::new("./model/mnist.json").exists() {
        println!("Model exists, loading model from file.");
        net.load_weights("./model/mnist.json");
    }
    else {
        println!("Model does not exits, training from scratch...");

        net.train(&x_train,
                  &y_train,
                  &TrainParameters::default()
                      .cost(Cost::MeanSquareError)
                      .learning_rate(0.5)
                      .learning_rate_decay(1.0)
                      .optimizer_sgd_momentum(0.9)
                      .l2(0.2)
                      .batch_size(64)
                      .iterations(Some(100000))
                      .log_interval(500)
                      .evaluation_dataset(Some((Tensor2D::NDArray(subset_x_predict),
                                                Tensor2D::NDArray(subset_y_predict))))
                      .target_stop_condition(Some(97.00)));

        net.save_weights("./model/mnist.json");
    }

    {
        let predictions = net.predict(&x_train);
        let y = y_train;
        println!("Dev Accuracy: {} %", NeuralNet::calculate_accuracy(&y, &predictions).to_string().bold());
        println!("Dev target {:?}", &y);
        println!("Dev predict {:?}", &predictions);
    }
    {
        let predictions = net.predict(&x_predict);
        let y = y_predict;
        println!("Test Accuracy: {} %", NeuralNet::calculate_accuracy(&y, &predictions).to_string().bold());
    }

}



fn load_mnist_dataset(testing: bool) -> (ndarray::Array2<f32>, ndarray::Array2<f32>) {
    // Gets testing dataset.
    let (images, labels): (Vec<f32>, Vec<f32>) = if testing {
        (
            mnist_read::read_data("data/mnist/t10k-images-idx3-ubyte")
                .into_iter()
                .map(|d| d as f32 / 255f32)
                .collect(),
            mnist_read::read_labels("data/mnist/t10k-labels-idx1-ubyte")
                .into_iter()
                .map(|l| l as f32)
                .collect(),
        )
    }
    // Gets training dataset.
    else {
        (
            mnist_read::read_data("data/mnist/train-images-idx3-ubyte")
                .into_iter()
                .map(|d| d as f32 / 255f32)
                .collect(),
            mnist_read::read_labels("data/mnist/train-labels-idx1-ubyte")
                .into_iter()
                .map(|l| l as f32)
                .collect(),
        )
    };
    let img_size = 28 * 28;

    // convert to one hot encoding

    let mut one_hot_encoded_labels = vec![];
    for label in labels.into_iter() {
        if  label == 0. {
            let encoding: [f32; 10] = [
                1.,0.,0.,0.,0.,0.,0.,0.,0.,0.
            ];
            one_hot_encoded_labels.push(encoding)
        }
        else if label == 1. {
            let encoding: [f32; 10] = [
                0.,1.,0.,0.,0.,0.,0.,0.,0.,0.
            ];
            one_hot_encoded_labels.push(encoding)
        }
        else if label == 2. {
            let encoding: [f32; 10] = [
                0.,0.,1.,0.,0.,0.,0.,0.,0.,0.
            ];
            one_hot_encoded_labels.push(encoding)
        }
        else if label == 3. {
            let encoding: [f32; 10] = [
                0.,0.,0.,1.,0.,0.,0.,0.,0.,0.
            ];
            one_hot_encoded_labels.push(encoding)
        }
        else if label == 4. {
            let encoding: [f32; 10] = [
                0.,0.,0.,0.,1.,0.,0.,0.,0.,0.
            ];
            one_hot_encoded_labels.push(encoding)
        }
        else if label == 5. {
            let encoding: [f32; 10] = [
                0.,0.,0.,0.,0.,1.,0.,0.,0.,0.
            ];
            one_hot_encoded_labels.push(encoding)
        }
        else if label == 6. {
            let encoding: [f32; 10] = [
                0.,0.,0.,0.,0.,0.,1.,0.,0.,0.
            ];
            one_hot_encoded_labels.push(encoding)
        }
        else if label == 7. {
            let encoding: [f32; 10] = [
                0.,0.,0.,0.,0.,0.,0.,1.,0.,0.
            ];
            one_hot_encoded_labels.push(encoding)
        }
        else if label == 8. {
            let encoding: [f32; 10] = [
                0.,0.,0.,0.,0.,0.,0.,0.,1.,0.
            ];
            one_hot_encoded_labels.push(encoding)
        }
        else if label == 9. {
            let encoding: [f32; 10] = [
                0.,0.,0.,0.,0.,0.,0.,0.,0.,1.
            ];
            one_hot_encoded_labels.push(encoding)
        }
    }
    println!("SIZE: {}", &one_hot_encoded_labels.len());
    let encoded_label = arr2(one_hot_encoded_labels[..].try_into().unwrap());
    return (
        ndarray::Array::from_shape_vec((images.len() / img_size, img_size), images).expect("Data shape wrong"),
        // ndarray::Array::from_shape_vec((labels.len(), 1), labels).expect("Label shape wrong"),
        encoded_label
    );

}