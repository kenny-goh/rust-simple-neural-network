#[allow(unused_imports)]
extern crate core;
pub mod rust_learn;

use crate::rust_learn::tensor2d::{Tensor2D};
use ndarray::{array, Array, Array2, Axis, arr1, s};

#[test]
fn test_neg_ndarray() {
    let m1 = Tensor2D::NDArray(array![[1.,1.]]);
    assert_eq!(-m1, Tensor2D::NDArray(array![[-1.,-1.]]));
}

#[test]
fn test_add_ndarray() {
    let m1 = Tensor2D::NDArray(array![[1.,1.]]);
    let m2 = Tensor2D::NDArray(array![[1.,1.]]);
    let m3 = m1 + m2;
    assert_eq!(m3, Tensor2D::NDArray(array![[2.,2.]]));
}

#[test]
fn test_add_ndarray_with_real_rhs() {
    let m1 = Tensor2D::NDArray(array![[1.,1.]]);
    let m2 = m1 + 1.;
    assert_eq!(m2, Tensor2D::NDArray(array![[2.,2.]]));
}

#[test]
fn test_add_ndarray_with_real_lhs() {
    let m1 = Tensor2D::NDArray(array![[1.,1.]]);
    let m2 = 1. + m1;
    assert_eq!(m2, Tensor2D::NDArray(array![[2.,2.]]));
}

#[test]
fn test_sub_ndarray() {
    let m1 = Tensor2D::NDArray(array![[1.,1.]]);
    let m2 = Tensor2D::NDArray(array![[1.,1.]]);
    let m3 = m1 - m2;
    assert_eq!(m3, Tensor2D::NDArray(array![[0.,0.]]));
}

#[test]
fn test_sub_ndarray_with_real_rhs() {
    let m1 = Tensor2D::NDArray(array![[1.,1.]]);
    let m2 = m1 - 1.;
    assert_eq!(m2, Tensor2D::NDArray(array![[0.,0.]]));
}

#[test]
fn test_sub_ndarray_with_real_lhs() {
    let m1 = Tensor2D::NDArray(array![[1.,1.]]);
    let m2 = 1. - m1;
    assert_eq!(m2, Tensor2D::NDArray(array![[0.,0.]]));
}

#[test]
fn test_mul_ndarray() {
    let m1 = Tensor2D::NDArray(array![[1.,1.]]);
    let m2 = Tensor2D::NDArray(array![[2.,2.]]);
    let m3 = m1 * m2;
    assert_eq!(m3, Tensor2D::NDArray(array![[2.,2.]]));
}

#[test]
fn test_mul_ndarray_with_real_lhs() {
    let m1 = Tensor2D::NDArray(array![[1.,1.]]);
    let m3 = m1 * 2.0;
    assert_eq!(m3, Tensor2D::NDArray(array![[2.,2.]]));
}

#[test]
fn test_mul_ndarray_with_real_rhs() {
    let m1 = Tensor2D::NDArray(array![[1.,1.]]);
    let m3 = 2.0 * m1;
    assert_eq!(m3, Tensor2D::NDArray(array![[2.,2.]]));
}

#[test]
fn test_div_ndarray() {
    let m1 = Tensor2D::NDArray(array![[2.,2.]]);
    let m2 = Tensor2D::NDArray(array![[2.,2.]]);
    let m3 = m1 / m2;
    assert_eq!(m3, Tensor2D::NDArray(array![[1.,1.]]));
}

#[test]
fn test_div_ndarray_with_real_lhs() {
    let m1 = Tensor2D::NDArray(array![[2.,2.]]);
    let m3 = m1 / 2.0;
    assert_eq!(m3, Tensor2D::NDArray(array![[1.,1.]]));
}

#[test]
fn test_div_ndarray_with_real_rhs() {
    let m1 = Tensor2D::NDArray(array![[2.,2.]]);
    let m3 = 2.0 / m1;
    assert_eq!(m3, Tensor2D::NDArray(array![[1.,1.]]));
}

#[test]
fn test_dot_ndarray() {
    let m1 = Tensor2D::NDArray(array![[1.,1.]]);
    let m2 = Tensor2D::NDArray(array![[1.],[1.]]);
    let m3 = m1.dot(&m2);

    let a1 = array![[1.,1.]];
    let a2 = array![[1.],[1.]];
    let a3 = a1.dot(&a2);

    assert_eq!(m3.to_ndarray(), a3);
}

#[test]
fn test_sum() {
    let m1 = Tensor2D::NDArray(array![[1.,1.],[1., 1.]]);
    assert_eq!(m1.sum(), 4.);
}

#[test]
fn test_pow1() {
    let m1 = Tensor2D::NDArray(array![[2.,2.]]);
    assert_eq!(m1.powi(2), Tensor2D::NDArray(array![[4., 4.]]));
}

#[test]
fn test_exp() {
    let m1 = Tensor2D::NDArray(array![[2.,2.]]);
    assert_eq!(m1.exp(), Tensor2D::NDArray(array![[7.389056, 7.389056]]));
}


#[test]
fn test_transpose() {
    let m1 = Tensor2D::NDArray(array![[2.,2.,2.]]);
    assert_eq!(m1.t(), Tensor2D::NDArray(array![[2.],[2.],[2.]]));
}

#[test]
fn test_sigmoid() {
    let m1 = Tensor2D::NDArray(array![[10.]]);
    assert_eq!(m1.sigmoid(), Tensor2D::NDArray(array![[0.9999546]]));
}

#[test]
fn test_derivative_sigmoid() {
    let m1 = Tensor2D::NDArray(array![[10.]]);
    println!("{:?}", m1.derivative_sigmoid());
    let sigmoid = 1. / (1. + f32::exp(-10.0));
    let derivative_sigmoid = sigmoid * (1. - sigmoid);
    assert_eq!(m1.derivative_sigmoid(), Tensor2D::NDArray(array![[derivative_sigmoid]]));
}

#[test]
fn test_leaky_relu() {
    assert_eq!(Tensor2D::NDArray(array![[-5.]]).leaky_relu(), Tensor2D::NDArray(array![[0.01]]));
    assert_eq!(Tensor2D::NDArray(array![[10.]]).leaky_relu(), Tensor2D::NDArray(array![[10.]]));
}

#[test]
fn test_derivative_leaky_relu() {
    assert_eq!(Tensor2D::NDArray(array![[-5.]]).derivative_leaky_relu(), Tensor2D::NDArray(array![[0.01]]));
    assert_eq!(Tensor2D::NDArray(array![[10.]]).derivative_leaky_relu(), Tensor2D::NDArray(array![[1.]]));
}

#[test]
fn test_tanh() {
    let m1 = Tensor2D::NDArray(array![[10.]]);
    let n = 10.;
    let e = f32::exp(n);
    let neg_e = f32::exp(-n);
    let tanh = (e - &neg_e) / (e + &neg_e);
    assert_eq!(m1.tanh(), Tensor2D::NDArray(array![[tanh]]));
}

#[test]
fn test_derivative_tanh() {
    let m1 = Tensor2D::NDArray(array![[10.]]);
    let n = 10.;
    let derivative_tanh = 1.0 - (f32::powi(n, 2));
    assert_eq!(m1.derivative_tanh(), Tensor2D::NDArray(array![[derivative_tanh]]));
}

#[test]
fn test_slice() {
    let m1 =Tensor2D::NDArray(array![[1.,2.,3.],
                                 [1.,2.,3.]]);
    println!("{:?}", m1.cols(1,3));
}

#[test]
fn test_softmax() {
    let multiple = Tensor2D::NDArray(array![[5.,2.,1.,3.], [10.,0.,0.,0.]]);
    let result = multiple.softmax();
    println!("result {:?}", &result);
    //assert_eq!(Tensor2D::NDArray(array![[0.8309527, 0.041370697, 0.015219429, 0.112457216]]), result);
}

#[test]
fn test_softmax_derivative() {
    let multiple = Tensor2D::NDArray(array![[1.,2.], [1.,2.]]);
    println!("softmax derivative {:?}", &multiple.softmax_derivative());
}
