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

## todo
- NN Layer
- Adam algorithm
- Regularization
- GPU Support



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





3 LAYER
INIT W1 0.05709334823247868
INIT W2 0.014403596928805733
ITERATION 0 ****************************
LINEAR FORWARD TANH
Z1 51.512735216464556
A1 51.51225588991275
Z2 0.02442756620008749
A2 617.0061068915487
TANH Backward A1 51.51225588991275
dZ1 2.088557057436204
dW1 0.032796593640580025
TANH Backward A2 617.0061068915487
dZ2 145.00610689154857
dW2 0.030567341124472975
Cost after iteration 0: 145.00610689154857
ITERATION 1 ****************************
LINEAR FORWARD TANH
Z1 83.39924053776991
A1 83.3967416520091
Z2 144.98755384651477
A2 580.794752581258
TANH Backward A1 83.3967416520091
dZ1 1.7585201528338388
dW1 0.03534208550163671
dZ2 108.79475258125805
dW2 0.02241161018132145
TANH Backward A2 580.794752581258
Cost after iteration 1: 108.79475258125805
ITERATION 2 ****************************
LINEAR FORWARD TANH
Z1 45.377389152545554
A1 45.36884871924299
Z2 253.98374373443147
A2 553.7272746733253
TANH Backward A1 45.36884871924299
dZ1 0.5114723759711759
dW1 0.013244958623485658
dZ2 81.72727467332525
dW2 0.0317869062274062
TANH Backward A2 553.7272746733253
Cost after iteration 2: 81.72727467332525


NN
INIT W1 0.05709334823247868
INIT W2 0.014403596928805733
ITERATION 0 ****************************
LINEAR_FORWARD_TANH
Z1 51.512735216464556
A1 51.51225588991275
Z2 0.02442756620008749
A2 617.0061068915487
Tanh backward A1 51.51225588991275
dZ1 0.5221392635533907
dW1 0.00011006826801077529
Tanh backward A2 617.0061068915487
dZ2 36.25152668296111
dW2 0.014689540068095133
ITERATION 1 ****************************
Z1 83.39924048141702
LINEAR_FORWARD_TANH
A1 83.39674159565892
Z2 144.98755384655595
A2 580.7947525812477
Tanh backward A2 83.39674159565892
dZ2 108.79475258124779
dW2 0.02241161015849836
Tanh backward A1 3802.4360622072018
dZ1 1.7585201532691193
dW1 0.03534208549002114
Cost after iteration 1: 108.79475258124779
ITERATION 2 ****************************
Z1 45.37738918072407
LINEAR_FORWARD_TANH
A1 45.368848747407554
Z2 253.98374373430647
A2 553.7272746733561
Tanh backward A2 45.368848747407554
dZ2 81.72727467335622
dW2 0.03178690621383087
Tanh backward A1 3802.4360622072018
dZ1 0.5114728783624701
dW1 0.013244964931719297
Cost after iteration 2: 81.72727467335622
