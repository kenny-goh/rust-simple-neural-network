use std::ops::Neg;
use ndarray::{arr2, Array, array, Array2, ArrayView2, Axis, s};
use ndarray_stats::{
    errors::{EmptyInput, MinMaxError, QuantileError},
    interpolate::{Higher, Interpolate, Linear, Lower, Midpoint, Nearest},
    Quantile1dExt, QuantileExt,
};
pub struct NDArrayHelper {}

impl NDArrayHelper {


    pub fn sigmoid(Z: &Array2<f32>) -> Array2<f32> {
        1. / (1. + (Z.mapv(|x| (-x).exp())))
    }

    pub fn relu(Z: &Array2<f32>) -> Array2<f32> {
        Z.mapv(|x| f32::max(0., x))
    }

    pub fn tanh(Z: &Array2<f32>) -> Array2<f32> {
        let e = Z.mapv(|x| x.exp());
        let neg_e = Z.mapv(|x| (-x).exp());
        (&e - &neg_e) / (&e + &neg_e)
    }

    pub fn normalize(X: &Array2<f32>) -> Array2<f32> {
        let col_size = X.shape()[0];
        let row_size =  X.shape()[1];
        let mut rows: Vec<f32> = vec![];
        for i in 0..col_size {
            let temp = X.slice(s![i,..]).reversed_axes();
            let min = temp.min().unwrap();
            let max = temp.max().unwrap();
            let data = temp.iter().map(|x| ((*x)-min) / (max-min ) ).collect::<Vec<f32>>();
            rows.extend(data);
        }
        Array::from_shape_vec((col_size, row_size), rows).unwrap()
    }
}