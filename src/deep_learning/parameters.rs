use crate::deep_learning::costs::Cost;
use crate::deep_learning::optimizer::Optimizer;

macro_rules! builder_field {
    ($field:ident, $field_type:ty) => {
        pub fn $field(&mut self, $field: $field_type) -> &mut Self {
            self.$field = $field;
            self
        }
    }
}

pub struct TrainParameters {
    pub learning_rate: f32,
    pub iterations: Option<usize>,
    pub cost: Cost,
    pub log_interval: usize,
    pub save_snashot: bool,
    pub save_snapshot_interval: usize,
    pub batch_size: usize,
    pub optimizer: Optimizer,
    pub momentum: f32,
    pub learning_rate_decay: f32,
    pub l2: f32,
    pub target_stop_condition: Option<f32>,
    pub stop_no_improvement: usize,
    pub gradient_clipping: Option<(f32, f32)>
}

impl TrainParameters {
    pub fn default() -> TrainParameters {
        TrainParameters {
            learning_rate: 1.0,
            log_interval: 100,
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
            gradient_clipping: None
        }
    }

    builder_field!( learning_rate, f32);
    builder_field!( learning_rate_decay, f32);
    builder_field!( iterations, Option<usize>);
    builder_field!( cost, Cost);
    builder_field!( batch_size, usize);
    builder_field!( l2, f32);
    builder_field!( target_stop_condition, Option<f32>);
    builder_field!( stop_no_improvement, usize);
    builder_field!( gradient_clipping, Option<(f32, f32)>);

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


