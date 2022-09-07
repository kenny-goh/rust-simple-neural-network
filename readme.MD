# Simple neural network written in Rust

I have forgotten what I have learned about deep learning two years ago, so this is a curiosity project to relearn deep learning while learning Rust at the same time.

## Description
The goal of this project is to develop a deep feed-forward network from the ground up with Rust that 
is good enough to solve the [MNIST handwritten digits classification problem](https://en.wikipedia.org/wiki/MNIST_database) with decent accuracy (~98%) and take less than few minutes 
using CPU. The mnist is a dataset of handwritten digits used by the machine learning community for benchmarking machine learning
algorithms.

The mnist dataset consists of 60,000 instances of training data and 10,000 testing images.

Below is an example of some digits from the mnist training data.

![Mnist puzzle](./mnist_example.png)

The target is to create a generic feed-forward deep learning framework that can be configured to
predict problems such as the mnist handwritten digits by running training 
on training examples in order to learn to approximate the target output, in
this case whether the digit is a 1 to 9.

To achieve that, I plan to implement the followings:

- Minimal Keras like API with the following features:
  - [x] NN layers (Dense)
  - [x] Mini-batching
  - [x] Activation (Sigmoid, Tanh, Relu, Softmax ) 
  - Stochastic Gradient Descent Algorithm
    - [x] Vanilla
    - [x] Momentum
    - [x] RMSProps
    - Adams
  - Hyper-parameters
    - [x] Learning rate
    - [x] Decay learning rate
    - Regularization
      - [x] L2, 
      - [x] Dropout

- Stretch target:
  - Accelerate the training / prediction using GPU acceleration.

## Updates
- 06/09/22 - I am able to hit 98% accuracy on the mnist dataset after implementing softmax and fine-tuning the parameters to include 20% dropout probability. 

## Getting Started (running the mnist example)
- Make sure you have Rust installed. Please refer to https://www.rust-lang.org/
- Open your terminal
- Check out this project by using the command: git clone git@github.com:kenny-goh/rust-simple-neural-network.git
- Navigate to the project folder
- To run the demo, please follow the instructions below
```shell
Cargo build #This will build the project
Cargo run --release # This will run the simple neural network example training and predicting mnist digit recognition problem.
# The program will emit the following output
Model does not exits, training from scratch...
*****************************************
 Epoch: 0
*****************************************
Batch 100   Dev Accuracy: 58.4  Eval Accuracy: 49.3  Costs: 0.668029
Batch 200   Dev Accuracy: 76.9  Eval Accuracy: 77.8  Costs: 0.483335
Batch 300   Dev Accuracy: 86.1  Eval Accuracy: 86.7  Costs: 0.332205
Batch 400   Dev Accuracy: 87.6  Eval Accuracy: 88.2  Costs: 0.291948
Batch 500   Dev Accuracy: 89.2  Eval Accuracy: 89.7  Costs: 0.247547
Batch 600   Dev Accuracy: 87.6  Eval Accuracy: 89.9  Costs: 0.276564
Batch 700   Dev Accuracy: 89.2  Eval Accuracy: 89.6  Costs: 0.219311
Batch 800   Dev Accuracy: 80    Eval Accuracy: 90.2  Costs: 0.306882
Batch 900   Dev Accuracy: 93.8  Eval Accuracy: 91.5  Costs: 0.132207
Total time in millis: 13880
Number of epoch: 0
Number of iterations: 938
Dev Accuracy: 93.84615 %
*****************************************
 Epoch: 1
*****************************************
Batch 62    Dev Accuracy: 100   Eval Accuracy: 92    Costs: 0.063227
Batch 162   Dev Accuracy: 96.9  Eval Accuracy: 92.7  Costs: 0.135550
Batch 262   Dev Accuracy: 92.3  Eval Accuracy: 92.9  Costs: 0.173187
Batch 362   Dev Accuracy: 95.3  Eval Accuracy: 93.1  Costs: 0.139057
Batch 462   Dev Accuracy: 93.8  Eval Accuracy: 94    Costs: 0.163107
Batch 562   Dev Accuracy: 96.9  Eval Accuracy: 93.8  Costs: 0.137667
Batch 662   Dev Accuracy: 90.7  Eval Accuracy: 93.7  Costs: 0.281303
Batch 762   Dev Accuracy: 98.4  Eval Accuracy: 94.6  Costs: 0.072723
Batch 862   Dev Accuracy: 95.3  Eval Accuracy: 94.2  Costs: 0.114708
Total time in millis: 27449
Number of epoch: 1
Number of iterations: 1876
Dev Accuracy: 95.38461 %
*****************************************
 Epoch: 2
*****************************************
Batch 24    Dev Accuracy: 95.3  Eval Accuracy: 94.3  Costs: 0.126417
Batch 124   Dev Accuracy: 92.3  Eval Accuracy: 94.5  Costs: 0.153767
Batch 224   Dev Accuracy: 92.3  Eval Accuracy: 94.6  Costs: 0.171772
Batch 324   Dev Accuracy: 92.3  Eval Accuracy: 95.8  Costs: 0.224238
Batch 424   Dev Accuracy: 96.9  Eval Accuracy: 94.7  Costs: 0.092061
Batch 524   Dev Accuracy: 92.3  Eval Accuracy: 95.3  Costs: 0.115753
Batch 624   Dev Accuracy: 96.9  Eval Accuracy: 95.6  Costs: 0.106947
Batch 724   Dev Accuracy: 84.6  Eval Accuracy: 95.2  Costs: 0.232226
Batch 824   Dev Accuracy: 93.8  Eval Accuracy: 95.5  Costs: 0.193509
Batch 924   Dev Accuracy: 100   Eval Accuracy: 95.2  Costs: 0.006374
Total time in millis: 41780
Number of epoch: 2
Number of iterations: 2814
Dev Accuracy: 100 %

....

*****************************************
 Epoch: 9
*****************************************
Batch 58    Dev Accuracy: 98.4  Eval Accuracy: 97.2  Costs: 0.038873
Batch 158   Dev Accuracy: 98.4  Eval Accuracy: 97.6  Costs: 0.033776
Batch 258   Dev Accuracy: 100   Eval Accuracy: 97.5  Costs: 0.025604
Batch 358   Dev Accuracy: 100   Eval Accuracy: 97.7  Costs: 0.030431
Batch 458   Dev Accuracy: 98.4  Eval Accuracy: 97.2  Costs: 0.043888
Batch 558   Dev Accuracy: 100   Eval Accuracy: 97.5  Costs: 0.024426
Batch 658   Dev Accuracy: 100   Eval Accuracy: 98    Costs: 0.024856

********* Halt due target stop condition met ************
Total time in millis: 134647
Number of epoch: 9
Number of iterations: 9100
Dev Accuracy: 99.195 %
Test Accuracy: 98.049995 %
```


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


## Authors
Kenny Goh

## Version History
* 0.1
    * Initial Release

## License
This project is licensed under the MIT License 




