use std::{fmt, ops};
use std::ops::{Add, Div, Mul, Neg, Sub};
use ndarray::{array, Array, Array2, Axis};
use ndarray_rand::RandomExt;
use serde::{Serialize, Deserialize};
use rand::distributions::Uniform;

pub enum RandomWeightInitStrategy {
    Random,
    Xavier
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Tensor2D {
    NDArray(Array2<f32>)
}

impl Tensor2D {
    
    pub fn init_random_as_ndarray(cols: usize, rows: usize, strategy: &RandomWeightInitStrategy) ->Tensor2D {
        match strategy {
            RandomWeightInitStrategy::Random => {
                let random_weights = Array::random((cols, rows), Uniform::new(0., 1.)) * 0.001;
                Tensor2D::NDArray(random_weights)
            }
            RandomWeightInitStrategy::Xavier => {
                let lower = -1. / f32::sqrt(rows as f32);
                let upper = 1. / f32::sqrt(rows as f32);
                let random_weights = Array::random((cols, rows), Uniform::new(0., 1.));
                let xavier_weights = lower + random_weights * (upper - lower);
                Tensor2D::NDArray(xavier_weights)
            }
        }
    }

    pub fn init_zeros_as_ndarray(cols: usize) ->Tensor2D {
        Tensor2D::NDArray(Array2::zeros((cols, 1)))
    }
    
    pub fn to_raw_vec(&self) -> Vec<f32> {
        match self {
            Tensor2D::NDArray(array) => {
                array.clone().into_raw_vec()
            }
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor2D::NDArray(array) => {
                array.shape()
            }
        }
    }

    pub fn size(&self, axis: usize) -> f32 {
        match self {
            Tensor2D::NDArray(array) => {
                array.len_of(Axis(axis)) as f32
            }
        }
    }

    pub fn dot(&self, rhs: &Tensor2D) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let other_result = match rhs {
                    Tensor2D::NDArray(array) => { array }
                };
                Tensor2D::NDArray(array.dot(other_result))
            }
        }
    }

    pub fn t(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.t().to_owned())
            }
        }
    }

    pub fn ln(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|x| f32::ln(x)))
            }
        }
    }

    pub fn powi(&self, i: i32) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|x| f32::powi(x, i)))
            }
        }
    }

    pub fn exp(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|x| f32::exp(x)))
            }
        }
    }

    pub fn sum(&self) -> f32 {
        match self {
            Tensor2D::NDArray(array) => {
                array.sum()
            }
        }
    }

    pub fn sigmoid(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(1. / (1. + (array.mapv(|x| (-x).exp()))))
            }
        }
    }

    pub fn derivative_sigmoid(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(_) => {
                self.sigmoid() * (1. - self.sigmoid())
            }
        }
    }

    pub fn leaky_relu(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|x| f32::max(0.01, x)))
            }
        }
    }

    pub fn derivative_leaky_relu(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|x| if x < 0. { 0.01 } else { 1. }))
            }
        }
    }

    pub fn tanh(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let e = array.mapv(|x| x.exp());
                let neg_e = array.mapv(|x| (-x).exp());
                Tensor2D::NDArray((&e - &neg_e) / (&e + &neg_e))
            }
        }
    }

    pub fn derivative_tanh(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(1.0 - (array.mapv(|x| x.powi(2))))
            }
        }
    }

    pub fn to_binary_value(&self, treshold: f32) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|a| if a > treshold { 1.0 } else { 0.0 }))
            }
        }
    }

    pub fn to_ndarray(&self) -> &Array2<f32> {
        return if let Tensor2D::NDArray(array) = self { &array } else {
            panic!("Not a closure object")
        };
    }

    pub fn is_ndarray(&self) -> bool {
        return match self {
            Tensor2D::NDArray(_) => { true }
            _ => { false }
        };
    }
}

impl Neg for Tensor2D {
    type Output = Self;
    fn neg(self) -> Self::Output {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(-array)
            }
        }
    }
}

impl Add<Tensor2D> for Tensor2D {
    type Output = Tensor2D;
    fn add(self, rhs: Tensor2D) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let array2 = match rhs {
                    Tensor2D::NDArray(array) => { array }
                };
                Tensor2D::NDArray(array + array2)
            }
        }
    }
}


impl Add<&Tensor2D> for &Tensor2D {
    type Output = Tensor2D;
    fn add(self, rhs: &Tensor2D) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let other_result = match rhs {
                    Tensor2D::NDArray(array) => { array }
                };
                Tensor2D::NDArray(array + other_result)
            }
        }
    }
}

impl Add<f32> for Tensor2D {
    type Output = Tensor2D;
    fn add(self, rhs: f32) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array + rhs)
            }
        }
    }
}

impl Add<Tensor2D> for f32 {
    type Output = Tensor2D;
    fn add(self, rhs: Tensor2D) -> Tensor2D {
        match rhs {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(self + array)
            }
        }
    }
}


impl Sub<f32> for Tensor2D {
    type Output = Tensor2D;
    fn sub(self, rhs: f32) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array - rhs)
            }
        }
    }
}

impl Sub<Tensor2D> for Tensor2D {
    type Output = Tensor2D;
    fn sub(self, rhs: Tensor2D) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let other_result = match rhs {
                    Tensor2D::NDArray(array) => { array }
                };
                Tensor2D::NDArray(array - &other_result)
            }
        }
    }
}

impl Sub<&Tensor2D> for &Tensor2D {
    type Output = Tensor2D;
    fn sub(self, rhs: &Tensor2D) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let other_result = match rhs {
                    Tensor2D::NDArray(array) => { array }
                };
                Tensor2D::NDArray(array - other_result)
            }
        }
    }
}


impl Sub<Tensor2D> for f32 {
    type Output = Tensor2D;
    fn sub(self, rhs: Tensor2D) -> Tensor2D {
        match rhs {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(self - array)
            }
        }
    }
}

impl Sub<&Tensor2D> for f32 {
    type Output = Tensor2D;
    fn sub(self, rhs: &Tensor2D) -> Tensor2D {
        match rhs {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(self - array)
            }
        }
    }
}


impl Mul<Tensor2D> for Tensor2D {
    type Output = Tensor2D;
    fn mul(self, rhs: Tensor2D) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let other_result = match rhs {
                    Tensor2D::NDArray(array) => { array }
                };
                Tensor2D::NDArray(array * &other_result)
            }
        }
    }
}

impl Mul<&Tensor2D> for Tensor2D {
    type Output = Tensor2D;
    fn mul(self, rhs: &Tensor2D) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let other_result = match rhs {
                    Tensor2D::NDArray(array) => { array }
                };
                Tensor2D::NDArray(array * other_result)
            }
        }
    }
}

impl Mul<f32> for Tensor2D {
    type Output = Tensor2D;
    fn mul(self, rhs: f32) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array * rhs)
            }
        }
    }
}

impl Mul<f32> for &Tensor2D {
    type Output = Tensor2D;
    fn mul(self, rhs: f32) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array * rhs)
            }
        }
    }
}


impl Mul<Tensor2D> for f32 {
    type Output = Tensor2D;
    fn mul(self, rhs: Tensor2D) -> Tensor2D {
        match rhs {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(self * array)
            }
        }
    }
}

impl Div<Tensor2D> for Tensor2D {
    type Output = Tensor2D;
    fn div(self, rhs: Tensor2D) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let other_result = match rhs {
                    Tensor2D::NDArray(array) => { array }
                };
                Tensor2D::NDArray(array / &other_result)
            }
        }
    }
}

impl Div<&Tensor2D> for &Tensor2D {
    type Output = Tensor2D;
    fn div(self, rhs: &Tensor2D) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let other_result = match rhs {
                    Tensor2D::NDArray(array) => { array }
                };
                Tensor2D::NDArray(array / other_result)
            }
        }
    }
}

impl Div<f32> for Tensor2D {
    type Output = Tensor2D;
    fn div(self, rhs: f32) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array / rhs)
            }
        }
    }
}

impl Div<Tensor2D> for f32 {
    type Output = Tensor2D;
    fn div(self, rhs: Tensor2D) -> Tensor2D {
        match rhs {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(self / array)
            }
        }
    }
}

impl Div<&Tensor2D> for f32 {
    type Output = Tensor2D;
    fn div(self, rhs: &Tensor2D) -> Tensor2D {
        match rhs {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(self / array)
            }
        }
    }
}



impl PartialEq for Tensor2D {
    fn eq(&self, other: &Self) -> bool {
        match (&self, &other) {
            (Tensor2D::NDArray(a), Tensor2D::NDArray(b)) => a == b
        }
    }
}

impl Clone for Tensor2D {
    fn clone(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.clone())
            }
        }
    }
}

impl fmt::Display for Tensor2D {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            Tensor2D::NDArray(array) => {
                write!(f, "{}", &array)
            }
        }
    }
}

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
fn test_ln() {
    let m1 = Tensor2D::NDArray(array![[2.,2.],[2., 2.]]);
    assert_eq!(m1.ln(), Tensor2D::NDArray(array![[0.6931472, 0.6931472],[0.6931472, 0.6931472]]));
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