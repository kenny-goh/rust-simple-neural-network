use crate::deep_learning::tensor2d::Tensor2D;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    LeakRelu,
    Tanh,
    Sigmoid,
}

impl Activation {
    pub fn compute(&self, z: &Tensor2D) -> Tensor2D {
        return match self {
            Self::Sigmoid => z.sigmoid(),
            Self::Tanh => z.tanh(),
            Self::LeakRelu => z.leaky_relu(),
        };
    }
    pub fn compute_derivative(&self, a: Tensor2D) -> Tensor2D {
        return match self {
            Self::Sigmoid => a.derivative_sigmoid(),
            Self::Tanh => a.derivative_tanh(),
            Self::LeakRelu => a.derivative_leaky_relu(),
        };
    }
}
