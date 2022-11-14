use std::ops::Deref;

pub mod mat;
pub mod num;
pub mod net;
pub mod cost;
pub mod num_fn;
pub mod activation;
pub mod mnist;

use mnist::Reader;
use net::Network;

use std::io::Cursor;
use image::io::Reader as ImageReader;

fn main() {
    Network::new([5, 10, 5, 6, 50]);

    let reader = Reader::new(Some(7), true);
}