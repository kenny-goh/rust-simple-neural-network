# Simple neural network written in Rust

A simple implementation of a neural network from scratch in Rust, based on my recollection of what I have learned from Andrew Ng's Deep Learning course on Coursera
two years ago.

## Description


## Getting Started
- Make sure you have Rust installed. Please refer to https://www.rust-lang.org/
- Open your terminal
- Check out this project by using the command: git clone git@github.com:kenny-goh/rust-simple-neural-network.git
- Navigate to the project folder
- To run the demo, please follow the instructions below
```shell
Cargo build #This will build the project
Cargo run --release # This will run the simple neural network 
# The program will emit the following output
....
*****************************************
 Epoch: 0
*****************************************
Batch 1     Accuracy: 50.3  Costs: 0.887732
Batch 2     Accuracy: 63.2  Costs: 0.749341
Batch 3     Accuracy: 68.1  Costs: 0.681430
....
....
Batch 39    Accuracy: 99.8  Costs: 0.054997

********* Halt due target stop condition met ************

Number of epoch: 6
Number of iterations: 259

Dev Accuracy: 100 %
Test Accuracy: 100 %
```


## Progress
- Type
  - [x] Sequential
  - RNN
  - Convolution

- [x] Minibatch
- Weight init
  - [x] Random
  - [x] Xavier
- Layer 
  - [x] Dense Layer
    - [x] Forward props
    - [x] Backward props
  - Flatten
- Activation
  - [x] Tanh
  - [x] Leaky Relu
  - [x] Sigmoid
  - SoftMax
- Tensor2D
  - Normalise
  - Shuffle
  - Operator overloading
    - [x] Add
    - [x] Sub
    - [x] Mul
    - [x] Div
  - [x] Tanh (and derivative)
  - [x] Sigmoid (and derivative)
  - [x] Relu (and derivative)
  - [x] LeakyRelu (and derivative)
  - [x] Softmax (and derivative)
- Serialization
  - [x] Can serialize and deserialize weights
  - Can serialize and deserialize architecture + weight
- Optimizer algorithm:
  - [x] SGD
  - [x] SGD with momentum
  - [x] RMS Props
  - Adam
- Regularization
  - L2
  - Drop Out Layer
  - Batch Norm
- GPU support
  [x]Refactor the code so that NDArray is one of the available 'backend' for linear algebra and vector operations
  - Add support for Firebase or WebGPU backend
- Testing
  - Minst dataset


## Authors
Kenny Goh

## Version History
* 0.1
    * Initial Release

## License

This project is licensed under the MIT License 


otal time in millis: 776994
Number of epoch: 53
Number of iterations: 50001
Dev Accuracy: 89.9 %
Dev Accuracy: 90.62 %
Test Accuracy: 90.72 %


