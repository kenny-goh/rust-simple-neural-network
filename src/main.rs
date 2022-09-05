#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unused_assignments)]
#![allow(unused_mut)]
extern crate core;

use colored::*;
mod rust_learn;
mod examples;

use crate::examples::banknote::bank_note_auth_example;
use crate::examples::login_bot_detector::login_bot_detector;
use crate::examples::checkout_bot_detector::checkout_bot_detector;
use crate::examples::mnist::mnist_example;

fn main() {
       mnist_example();
       // checkout_bot_detector();
       // bank_note_auth_example();
       // login_bot_detector();
}

