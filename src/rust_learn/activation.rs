use crate::rust_learn::tensor2d::Tensor2D;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    LeakyRelu,
    Relu,
    Tanh,
    Sigmoid,
    SoftMax,
}

impl Activation {
    pub fn compute(&self, z: &Tensor2D) -> Tensor2D {
        return match self {
            Self::Sigmoid => z.sigmoid(),
            Self::Tanh => z.tanh(),
            Self::LeakyRelu => z.leaky_relu(),
            Self::SoftMax => z.softmax(),
            Self::Relu => z.relu()
        };
    }
    pub fn compute_derivative(&self, a: Tensor2D) -> Tensor2D {
        return match self {
            Self::Sigmoid => a.derivative_sigmoid(),
            Self::Tanh => a.derivative_tanh(),
            Self::Relu => a.relu(),
            Self::LeakyRelu => a.derivative_leaky_relu(),
            Self::SoftMax => a.softmax_derivative(),
        };
    }
}
