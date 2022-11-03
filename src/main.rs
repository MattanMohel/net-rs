use std::ops::Deref;

pub mod mat;
pub mod num;
pub mod net;
pub mod act;
pub mod cost;

// use net::Net;

use mat::*;
use net::Net;

fn main() {
    let input = Matrix::from_arr([[0.23, 0.11, 0.76]]).transpose();
    let expected = Matrix::from_arr([[0.0, 0.0, 0.0]]).transpose();

    let mut net = Net::new([3, 2, 3]);

    net.train(&input, &expected);
}