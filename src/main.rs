use ndarray::prelude::*;

mod algorithm;
mod examples;
mod utils;

use crate::examples::banknote::bank_note_auth_example;
use crate::examples::xor::xor_example;

fn main() {
    // xor_example();
    bank_note_auth_example();
}


