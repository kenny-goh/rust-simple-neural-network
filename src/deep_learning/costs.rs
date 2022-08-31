use ndarray::{Array, Axis};
use crate::deep_learning::tensor2d::Tensor2D;

#[derive(Debug)]
pub enum Cost {
    CrossEntropy,
    Quadratic,
}

impl Cost {
    pub fn cost(&self, activation: &Tensor2D, target: &Tensor2D) -> f32 {
        return match self {
            Self::CrossEntropy => {
                let m = target.size(1);
                let a = activation.ln() * target;
                let b = (1. - target) * (1. - activation).ln();
                let log_probs = a + b;
                let cost = log_probs.sum() * (-1.0 / m);
                if f32::is_nan(cost) { 0. } else { cost }
            }
            Self::Quadratic => {
                let m =  target.size(1);
                ((activation - target).powi(2)).sum() * 1. / m
            }
        };
    }
    pub fn derivative(&self, y: &Tensor2D, al: &Tensor2D) -> Tensor2D {
        return match self {
            Self::CrossEntropy => {
                -(y / al - (1.0 - y) / (1.0 - al))
            }
            Self::Quadratic => {
                al - y
            }
        };
    }
}
