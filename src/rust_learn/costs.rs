use crate::rust_learn::tensor2d::Tensor2D;
use colored::*;

#[derive(Debug)]

//
pub enum Cost {
    CrossEntropy,
    MeanSquareError,
}

impl Cost {
    pub fn cost(&self, activation: &Tensor2D, target: &Tensor2D) -> f32 {
        return match self {
            Self::CrossEntropy => {
                let m = target.size(1);
                let a = activation.log() * target;
                let b = (1. - target + 1e-20) * (1. - activation + 1e-20).log();
                let log_probs = &a + &b;
                let mut cost = &log_probs.sum() * (-1. / m);
                if f32::is_nan(cost) {
                    panic!("{}","OVERFLOW/UNDERFLOW detected. Please checking your training parameters or data".red());
                }
                cost
            }
            Self::MeanSquareError => {
                let m =  target.size(1);
                ((activation - target).powi(2)).sum() * 1. / m
            }
        };
    }
    pub fn derivative(&self, y: &Tensor2D, al: &Tensor2D) -> Tensor2D {
        return match self {
            Self::CrossEntropy => {
                // -(y / al - (1. - y) / (1. - al))
                (&(y * -1f32) / al + 1e-20) + (1. - y) / ((1. - al) + 1e-20)
            }
            Self::MeanSquareError => {
                al - y
            }
        };
    }
}
