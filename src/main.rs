use std::ops::Deref;

pub mod mat;
pub mod num;
pub mod net;
pub mod act;
pub mod cost;

use num::N;
use mat::*;
use net::Net;

fn main() {
    Net::new([5, 10, 5, 6, 50]);
}