# Simple neural network written in Rust

A simple implementation of a neural network from scratch in Rust, based on my recollection of what I have learned from Andrew Ng's Deep Learning course on Coursera
two years ago.

## Description


## Getting Started
- Make sure you have Rust installed. Please refer to https://www.rust-lang.org/
- Open your terminal
- Check out this project by using the command: git clone git@github.com:kenny-goh/rust-simple-neural-network.git
- Navigate to the project folder
```shell
Cargo build #This will build the project
Cargo run --release # This will run the simple neural network 
# The program will emit the following output
Cost after iteration 0: 145.00539107115316
...
...
...
Cost after iteration 49900: 8.074419724594703
Accuracy: 99.67611336032388 %
```

## todo
## Progress
- MVP 
  - [x] Supports Dense Layer 
  - [x] Activation: LeakyRelu
  - [x] Activation: Tanh
  - [x] Activation: Sigmoid
  - [x] Forward props **[ndarray]**
  - [x] Backward props **[ndarray]**
- Testing
  - Minst dataset
- Can serialize and deserialize
- Algorithm: 
  - [x] SGD
  - Adam algorithm
- Regularization
  - L2
  - Drop Out Layer
- GPU support
  - Refactor the code so that NDArray is one of the available 'backend' for linear algebra and vector operations
  - Add support for Firebase backend

  

## Authors
Kenny Goh

## Version History
* 0.1
    * Initial Release

## License

This project is licensed under the MIT License 




