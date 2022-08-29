#[allow(dead_code, unused_imports)]
mod algorithm;
mod examples;
mod utils;

use crate::examples::banknote::bank_note_auth_example;
use crate::examples::xor::xor_example;
use crate::examples::login_bot_detector::login_bot_detector;
use crate::examples::checkout_bot_detector::checkout_bot_detector;

fn main() {
    // xor_example();
    // checkout_bot_detector();
     bank_note_auth_example();
    // login_bot_detector();

}

