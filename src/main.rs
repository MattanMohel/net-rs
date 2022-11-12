use std::ops::Deref;

pub mod mat;
pub mod num;
pub mod net;
pub mod cost;
pub mod num_fn;
pub mod activation;

use net::Network;

fn main() {
    Network::new([5, 10, 5, 6, 50]);
}