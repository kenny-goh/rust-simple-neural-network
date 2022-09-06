# Simple neural network written in Rust

I have forgotten what I have learned about deep learning two years ago, so this is a curiosity project to relearn deep learning while learning Rust at the same time.

## Description
The goal of this project is to simply develop a feed-forward network (MLP) from the ground up that 
is good enough to solve the MNIST classification problem using Rust.

At the bare minimum, the library will implement the following features:

- Minimalistic KERAS like implementation that can solve MNIST digit classification problem. Should support forward based network with the following featuers:
  - [x] NN layers (Dense)
  - [x] Mini-batching
  - [x] Activation (Sigmoid, Tanh, Relu ) 
  - SGD 
    - [x] Vanilla
    - [x] Momentum
    - [x] RMSProps
    - Adams
  - Hyper-parameters
    - [x] Learning rate
    - [x] Decay learning rate
    - Regularization
      - [x] L2, 
      - Dropout
  - Can train up to 97% accuracy on MNIST dataset

- Stretch target:
  - Accelerate the training / prediction using GPU acceleration.
  - Softmax activation function (Derivative of softmax is tricky to implement with NDArray)
  - Can train up to 98% accuracy on MNIST dataset

## Toy example to solve XOR problem
```rust

 let mut net = NeuralNet::new(2,
    &[MetaLayer::Dense(10, Activation::LeakyRelu), 
      MetaLayer::Dense(1, Activation::Sigmoid)], 
    &WeightInitStrategy::Xavier);

    let x_predict = Tensor2D::NDArray(array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]].reversed_axes());
    let y_predict = Tensor2D::NDArray(array![[0.], [1.], [1.], [0.]].reversed_axes());

      
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

    let predictions = net.predict(&x_predict);
    println!("Test Accuracy: {} %", NeuralNet::calculate_accuracy(&y, &predictions).to_string().bold());

```

## Getting Started
- Make sure you have Rust installed. Please refer to https://www.rust-lang.org/
- Open your terminal
- Check out this project by using the command: git clone git@github.com:kenny-goh/rust-simple-neural-network.git
- Navigate to the project folder
- To run the demo, please follow the instructions below
```shell
Cargo build #This will build the project
Cargo run --release # This will run the simple neural network example training and predicting mnist digit recognition problem.
# The program will emit the following output
*****************************************
 Epoch: 0
*****************************************
Batch 500   Test Accuracy: 90    Costs: 0.149351
Total time in millis: 16009
Number of epoch: 0
Number of iterations: 938
Dev Accuracy: 90 %
*****************************************
 Epoch: 1
*****************************************
Batch 62    Test Accuracy: 95.6  Costs: 0.037166
Batch 562   Test Accuracy: 95    Costs: 0.067823
Total time in millis: 32375
Number of epoch: 1
Number of iterations: 1876
Dev Accuracy: 95 %
*****************************************
 Epoch: 2
*****************************************
Batch 124   Test Accuracy: 96    Costs: 0.067899
Batch 624   Test Accuracy: 95.8  Costs: 0.023466
Total time in millis: 49012
Number of epoch: 2
Number of iterations: 2814
Dev Accuracy: 95.8 %
*****************************************
 Epoch: 3
*****************************************
Batch 186   Test Accuracy: 95.8  Costs: 0.114590
Batch 686   Test Accuracy: 96.6  Costs: 0.044832
Total time in millis: 65355
Number of epoch: 3
Number of iterations: 3752
Dev Accuracy: 96.6 %
*****************************************
 Epoch: 4
*****************************************
Batch 248   Test Accuracy: 96.6  Costs: 0.045624
Batch 748   Test Accuracy: 96.8  Costs: 0.000666
Total time in millis: 81910
Number of epoch: 4
Number of iterations: 4690
```

## Authors
Kenny Goh

## Version History
* 0.1
    * Initial Release

## License

This project is licensed under the MIT License 




