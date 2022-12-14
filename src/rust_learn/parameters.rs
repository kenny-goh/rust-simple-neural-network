use crate::rust_learn::costs::Cost;
use crate::rust_learn::types::Optimizer;
use crate::rust_learn::tensor2d::Tensor2D;

macro_rules! builder_field {
    ($field:ident, $field_type:ty) => {
        pub fn $field(&mut self, $field: $field_type) -> &mut Self {
            self.$field = $field;
            self
        }
    }
}

/// Builder for training parameters
/// Ideally should be serializable into JSON
pub struct TrainParameters {
    pub learning_rate: f32,
    pub cost: Cost,
    pub log_interval: usize,
    pub save_snashot: bool,
    pub save_snapshot_interval: usize,
    pub batch_size: usize,
    pub optimizer: Optimizer,
    pub momentum: f32,
    pub learning_rate_decay: f32,
    pub l2: f32,
    pub stop_no_improvement: usize,
    pub iterations: Option<usize>,
    pub target_stop_condition: Option<f32>,
    pub gradient_clipping: Option<(f32, f32)>,
    pub evaluation_dataset: Option<(Tensor2D,Tensor2D)>,
}

impl TrainParameters {
    pub fn default() -> TrainParameters {
        TrainParameters {
            learning_rate: 1.0,
            log_interval: 5,
            cost: Cost::CrossEntropy,
            iterations: None,
            save_snashot: false,
            save_snapshot_interval: 0,
            batch_size: 32,
            optimizer: Optimizer::SGD,
            momentum: 0.9,
            learning_rate_decay: 0.0,
            l2: 0.0,
            target_stop_condition: Some(99.9),
            stop_no_improvement: 20,
            gradient_clipping: None,
            evaluation_dataset: None,
        }
    }

    builder_field!( log_interval, usize);
    builder_field!( learning_rate, f32);
    builder_field!( learning_rate_decay, f32);
    builder_field!( iterations, Option<usize>);
    builder_field!( cost, Cost);
    builder_field!( batch_size, usize);
    builder_field!( l2, f32);
    builder_field!( target_stop_condition, Option<f32>);
    builder_field!( stop_no_improvement, usize);
    builder_field!( gradient_clipping, Option<(f32, f32)>);
    builder_field!( evaluation_dataset, Option<(Tensor2D,Tensor2D)> );

    pub fn optimizer_sgd_momentum(&mut self, momentum: f32) -> &mut TrainParameters {
        self.optimizer = Optimizer::SGDMomentum;
        self.momentum = momentum;
        self
    }

    pub fn optimizer_rms_props(&mut self, momentum: f32) -> &mut TrainParameters {
        self.optimizer = Optimizer::RMSProps;
        self.momentum = momentum;
        self
    }
}


