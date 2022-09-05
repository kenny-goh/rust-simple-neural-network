use core::f32;
use std::{fmt};
use std::ops::{Add, Div, Mul, Neg, Sub};
use itertools::izip;
use ndarray::{arr1, arr2, Array, array, Array2, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use serde::{Serialize, Deserialize};
use rand::distributions::Uniform;

pub enum WeightInitStrategy {
    Random,
    Xavier
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Tensor2D {
    NDArray(Array2<f32>)
}


impl Tensor2D {
    pub fn ndarray_random_init(cols: usize, rows: usize, strategy: &WeightInitStrategy) ->Tensor2D {
        match strategy {
            WeightInitStrategy::Random => {
                let random_weights = Array::random((cols, rows), Uniform::new(0., 1.)) * 0.001;
                Tensor2D::NDArray(random_weights)
            }
            WeightInitStrategy::Xavier => {
                let lower = -1. / f32::sqrt(rows as f32);
                let upper = 1. / f32::sqrt(rows as f32);
                let random_weights = Array::random((cols, rows), Uniform::new(0., 1.));
                let xavier_weights = lower + random_weights * (upper - lower);
                Tensor2D::NDArray(xavier_weights)
            }
        }
    }

    pub fn ndarray_init_zeroes(cols: usize) ->Tensor2D {
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

    pub fn count_equal_rows(&self, other: &Tensor2D) ->f32 {
        let mut count = 0f32;
        match self { Tensor2D::NDArray(array) => {
                let array2 = other.to_ndarray();
                let rows = array2.shape()[1];
                for i in 0..rows {
                    let a1 = &array.slice(s![.., i]);
                    let a2 = &array2.slice(s![..,i]);
                    if a1 == a2 {
                        count += 1.;
                    }
                }
            }
        }
        count
    }

    pub fn log(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|x| f32::log10(x)))
            }
        }
    }

    pub fn sum_nan_all(&self, val: f32) -> f32 {
        match self {
            Tensor2D::NDArray(array) => {
                array.mapv(|x| if x.is_nan() { 0.} else { x }).sum()
            }
        }
    }

    pub fn cols(&self, from: i32, to: i32) ->Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let arr =  array.slice(s![..,from..to]);
                Tensor2D::NDArray(arr.to_owned())
            }
        }
    }

    pub fn clip(&self, min: f32, max: f32) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|x| {
                    if x < min {
                        return min
                    }
                    if x > max {
                        return max
                    }
                    x
                }))
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

    pub fn sqrt(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|x| f32::sqrt(x.max(1e-20))))
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

    pub fn sum_keep_dims(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(arr2(&[[array.sum()]]))
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

    pub fn relu(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|x| f32::max(0.0, x)))
            }
        }
    }

    pub fn softmax(&self)-> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let max = *array.max().unwrap();
                let shift_x = array - max;
                let exponents = shift_x.mapv(|x| f32::exp(x));
                let sum_of_exponents = exponents.sum();
                let softmax = exponents / sum_of_exponents;
                Tensor2D::NDArray(softmax)
            }
        }
    }

    // to implement:
    // def backward(self):
    // for i in range(len(self.value)):
    // for j in range(len(self.error)):
    // if i == j:
    // self.gradient[i] = self.value[i] * (1-self.input[i))
    // else:
    // self.gradient[i] = -self.value[i]*self.input[j]
    pub fn softmax_derivative(&self)->Tensor2D {
        // single formula to calculate the Jacobian derivative of the Softmax function is
        // np.diag(S) - (S_matrix * np.transpose(S_matrix))
        panic!("Not implemented")
    }

    pub fn derivative_leaky_relu(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|x| if x < 0. { 0.01 } else { 1. }))
            }
        }
    }

    pub fn derivative_relu(&self) -> Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(array.mapv(|x| if x < 0. { 0. } else { 1. }))
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
            panic!("Not an NDArray")
        };
    }

    pub fn is_ndarray(&self) -> bool {
        return match self {
            Tensor2D::NDArray(_) => { true }
        };
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


// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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
// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
impl Add<&Tensor2D> for f32 {
    type Output = Tensor2D;
    fn add(self, rhs: &Tensor2D) -> Tensor2D {
        match rhs {
            Tensor2D::NDArray(array) => {
                Tensor2D::NDArray(self + array)
            }
        }
    }
}

// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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
// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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
// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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
// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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

// fixme: Use Rust Macro to reduce boiler plate code
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
// fixme: Use Rust Macro to reduce boiler plate code
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

