# Simple neural network written in Rust

A simple implementation of a neural network from scratch in Rust, based on my recollection of what I have learned from Andrew Ng's Deep Learning course on Coursera
two years ago.

## Description
The application is a 3 layer neural network that reads in a sample banknote authentication data (example.txt)
with 4 features (X) and a binary output (Y) which is either 0 (no) and 1 (yes) classification.

The model consists of 3 layers
- Layer 1 (Input)
- Hidden Layer (Configured with 20 nodes)
- Layer 2 (Output)

The learning parameters are:
W1, W2 

When the model train against the training data, the weights (W1 and W2) will be incrementally
updated by the learning algorithm through back propagation.

After certain number of iterations has elapsed, the training stops and the program will try
to predict the Y output from the sample test data based on the learned parameters W1 and W2.

Once the prediction is done, the program will calculate how accurate is the prediction and print out the accuracy as an output. 

If all of these sounds alient to you, please have a look at this gentle introduction to Logistic regression at
https://iamtrask.github.io/2015/07/12/basic-python-network/

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

### Dependencies
* ndarray = "0.14.0"
* ndarray-rand = "0.13.0"
* rand = "0.8"
* rand_distr="0.4"

## Authors
Kenny Goh

## Version History
* 0.1
    * Initial Release

## License

This project is licensed under the MIT License 