use ndarray::{Array, Array2, s};
use ndarray_stats::{
    QuantileExt,
};

pub struct NDArrayHelper {}

impl NDArrayHelper {
    #[allow(dead_code)]
    pub fn normalize(x_input: &Array2<f32>) -> Array2<f32> {
        let col_size = x_input.shape()[0];
        let row_size =  x_input.shape()[1];
        let mut rows: Vec<f32> = vec![];
        for i in 0..col_size {
            let temp = x_input.slice(s![i,..]).reversed_axes();
            let min = temp.min().unwrap();
            let max = temp.max().unwrap();
            let data = temp.iter().map(|x| ((*x)-min) / (max-min ) ).collect::<Vec<f32>>();
            rows.extend(data);
        }
        Array::from_shape_vec((col_size, row_size), rows).unwrap()
    }
}