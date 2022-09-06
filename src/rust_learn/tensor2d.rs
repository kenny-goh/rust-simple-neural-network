use core::f32;
use std::{fmt};
use std::ops::{Add, Div, Mul, Neg, Sub};
use itertools::izip;
use ndarray::{arr1, arr2, Array, array, Array2, Axis, NdProducer, s};
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use serde::{Serialize, Deserialize};
use rand::distributions::Uniform;

macro_rules! unary_tensor_op {
    ($LhrOp:ty, $opname: ident, $lambda: expr) => {
        impl $LhrOp for Tensor2D {
            type Output = Self;
            fn $opname(self) -> Self::Output {
                match self {
                    Tensor2D::NDArray(array) => {
                        Tensor2D::NDArray($lambda(array))
                    }
                }
            }
        }
    }
}

macro_rules! tensor_op_tensor {
    ($op:ty, $target:ty, $rhs:ty, $opname: ident, $lambda: expr) => {
        impl $op for $target {
            type Output = Tensor2D;
            fn $opname(self, rhs: $rhs) -> Tensor2D {
                match self {
                    Tensor2D::NDArray(array) => {
                        let array2 = rhs.to_ndarray();
                        Tensor2D::NDArray($lambda(array, array2))
                    }
                }
            }
        }
    }
}

macro_rules! tensor_op_real {
    ($op:ty, $typ:ty, $target:ty, $opname: ident, $lambda: expr) => {
        impl $op for $target {
            type Output = Tensor2D;
            fn $opname(self, rhs: $typ) -> Tensor2D {
                match self {
                    Tensor2D::NDArray(array) => {
                        Tensor2D::NDArray($lambda(array, rhs))
                    }
                }
            }
        }
    }
}


macro_rules! real_op_tensor {
    ($op:ty, $target:ty, $opname: ident, $rhs: ty, $lambda: expr) => {
       impl $op for $target {
            type Output = Tensor2D;
            fn $opname(self, rhs: $rhs) -> Tensor2D {
                match rhs {
                    Tensor2D::NDArray(array) => {
                        Tensor2D::NDArray($lambda(self, array))
                    }
                }
            }
        }
    }
}


pub enum WeightInitStrategy {
    Random,
    Xavier
}

#[derive(Debug, Serialize, Deserialize)]
/// Tensor2D is a an abstract two dimensional matrix that implements common vector operations.
/// The intention of this abstraction is to provide at least two default backend vector implementations:
/// NDArray and probably WebGPU based vector.
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

    pub fn ndarray_random_mask(cols: usize, rows: usize, probs: f32) ->Tensor2D {
        let mut random = Array::random((cols, rows), Uniform::new(0., 1.));
        random = random.mapv(|x| if x >= probs {1.0} else {0.0} );
        Tensor2D::NDArray(random)
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
                Tensor2D::NDArray(array.mapv(|x| f32::log10(x.max( 1e-20))))
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
                let cols = array.shape()[0];
                let rows = array.shape()[1];

                // Maximum of each cols
                // e.g [[1,5,  ..]
                //      [2,10, ..]
                //      [3,2,  ..]]
                // -> [[3,10,..]]
                let max_of_each_rows = NDArrayUtils::max_axis_0(array);
                let max = NDArrayUtils::tile_rows(cols, rows, max_of_each_rows);

                // Shift x by subtracting max (for numerical stable softmax to allow large values)
                // e.g [[1-3,5-10,  ..]
                //      [2-3,10-10, ..]
                //      [3-3,2-10,  ..]]
                // ->  [[-2,-5,  ..]
                //      [-1,0, ..]
                //      [0,-8,  ..]]
                let shift_x = array - max;
                let exponents = shift_x.mapv(|x| f32::exp(x));

                // Sum of each cols
                // e.g [[1,5,  ..]
                //      [2,10, ..]
                //      [3,2,  ..]]
                // -> [[6,17,..]]
                let sum_of_exponents = exponents.sum_axis(Axis(0));

                let softmax = exponents /  sum_of_exponents;
                Tensor2D::NDArray(softmax)
            }
        }
    }

    pub fn softmax_derivative(&self)->Tensor2D {
        match self {
            Tensor2D::NDArray(array) => {
                let exponents = array.mapv(|x| f32::exp(x));
                let cols = array.shape()[0];
                let rows = array.shape()[1];

                // Sum of exponents - exponents
                let sums_of_exponents = exponents.sum_axis(Axis(0));
                let vec_sum_of_exponents = sums_of_exponents.clone().into_raw_vec();
                let exponent_sums_matrix: Array2<f32> = NDArrayUtils::tile_rows(cols, rows, vec_sum_of_exponents);
                let exponents_sums_sub =  exponent_sums_matrix - &exponents;

                // sums of exponents squared
                let sums_squared = &sums_of_exponents * &sums_of_exponents;
                let vec_sum_of_squared_exponents = sums_squared.clone().into_raw_vec();
                let exponent_sums_squared_matrix =  NDArrayUtils::tile_rows(cols, rows, vec_sum_of_squared_exponents);

                // dz = exponents * (sum of exponents - exponents / sum of exponents squared )
                let derivatives = &exponents * (exponents_sums_sub / (exponent_sums_squared_matrix));

                Tensor2D::NDArray(derivatives)
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


// macro to the rescue, this would have bloated to over 400 lines!

unary_tensor_op!(Neg, neg, |arr: Array2<f32> | -arr);

tensor_op_tensor!(Add<Tensor2D>, Tensor2D, Tensor2D, add, | arr: Array2<f32>, arr2: &Array2<f32> |  arr + arr2 );
tensor_op_tensor!(Add<&Tensor2D>, &Tensor2D, &Tensor2D, add, | arr: &Array2<f32>, arr2: &Array2<f32> |  arr + arr2 );
tensor_op_tensor!(Sub<Tensor2D>, Tensor2D, Tensor2D, sub, | arr: Array2<f32>, arr2: &Array2<f32> |  arr - arr2 );
tensor_op_tensor!(Sub<&Tensor2D>, &Tensor2D, &Tensor2D, sub, | arr: &Array2<f32>, arr2: &Array2<f32> |  arr - arr2 );
tensor_op_tensor!(Mul<Tensor2D>, Tensor2D, Tensor2D, mul, | arr: Array2<f32>, arr2: &Array2<f32> |  arr * arr2 );
tensor_op_tensor!(Mul<&Tensor2D>, &Tensor2D, &Tensor2D, mul, | arr: &Array2<f32>, arr2: &Array2<f32> |  arr * arr2 );
tensor_op_tensor!(Mul<&Tensor2D>, Tensor2D, &Tensor2D, mul, | arr: Array2<f32>, arr2: &Array2<f32> |  arr * arr2 );

tensor_op_tensor!(Div<Tensor2D>, Tensor2D, Tensor2D, div,| arr: Array2<f32>, arr2: &Array2<f32> |  arr / arr2 );
tensor_op_tensor!(Div<&Tensor2D>, &Tensor2D, &Tensor2D, div, | arr: &Array2<f32>, arr2: &Array2<f32> |  arr / arr2 );

tensor_op_real!(Add<f32>, f32, Tensor2D, add, | arr: Array2<f32>, arr2: f32 |  arr + arr2 );
tensor_op_real!(Sub<f32>, f32, Tensor2D, sub, | arr: Array2<f32>, arr2: f32 |  arr - arr2 );
tensor_op_real!(Mul<f32>, f32, Tensor2D, mul, | arr: Array2<f32>, arr2: f32 |  arr * arr2 );
tensor_op_real!(Mul<f32>, f32, &Tensor2D, mul, | arr: &Array2<f32>, arr2: f32 |  arr * arr2 );
tensor_op_real!(Div<f32>, f32, Tensor2D, div, | arr: Array2<f32>, arr2: f32 |  arr / arr2 );

real_op_tensor!(Add<Tensor2D>, f32, add, Tensor2D, | real: f32, arr: Array2<f32> |  real + arr );
real_op_tensor!(Add<&Tensor2D>, f32, add, &Tensor2D, | real: f32, arr: &Array2<f32> |  real + arr );
real_op_tensor!(Sub<Tensor2D>, f32, sub, Tensor2D, | real: f32, arr: Array2<f32> |  real - arr );
real_op_tensor!(Sub<&Tensor2D>, f32, sub, &Tensor2D, | real: f32, arr: &Array2<f32> |  real - arr );
real_op_tensor!(Mul<Tensor2D>, f32, mul, Tensor2D, | real: f32, arr: Array2<f32> |  real * arr );
real_op_tensor!(Mul<&Tensor2D>, f32, mul, &Tensor2D, | real: f32, arr: &Array2<f32> |  real * arr );
real_op_tensor!(Div<Tensor2D>, f32, div, Tensor2D, | real: f32, arr: Array2<f32> |  real / arr );
real_op_tensor!(Div<&Tensor2D>, f32, div, &Tensor2D, | real: f32, arr: &Array2<f32> |  real / arr );

struct NDArrayUtils {

}
impl NDArrayUtils {

    // Repeat columns (axis 0) N times vertically
    // cols: 3
    // rows: 4
    // vector: [1,2,3]
    // [1, 2, 3]
    // [1, 2, 3]
    // [1, 2, 3]
    // [1, 2, 3]
    fn tile_rows(cols: usize, rows: usize, vector: Vec<f32>) -> Array2<f32> {
        let vector_2d = Array2::from_shape_vec((rows, 1),
                                               vector).unwrap().reversed_axes();
        let ones: Array2<f32> = Array2::ones((cols, 1));
        ones.dot(&vector_2d)
    }

    // Move this to NDArray utils
    fn max_axis_0(array: &Array2<f32>) -> Vec<f32> {
        let rows = array.shape()[1];
        let mut max_of_each_rows: Vec<f32> = vec![];
        for i in 0..rows {
            let row = &array.slice(s![.., i]);
            let row_max = *row.max().unwrap();
            max_of_each_rows.push(row_max);
        }
        max_of_each_rows
    }
}
