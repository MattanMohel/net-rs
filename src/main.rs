use std::ops::Deref;

pub mod mat;
pub mod num;
pub mod net;
pub mod act;
pub mod cost;

use num::N;
use mat::*;
use net::Net;

fn pm(mat: &Matrix<N>) {
    for (i, n) in mat.iter().enumerate() {
        print!("{} ", n);

        if i % mat.col() == 0 {
            println!()
        }
    }
}

fn main() {
    let input = Matrix::from_arr([[0.23, 0.11, 0.76, 0.43, 0.01]]).transpose();
    let expected = Matrix::from_arr([[0.0, 0.54, 1.0]]).transpose();

    let mut net = Net::new([5, 10, 15, 10, 3]);

    for i in 0..100 {    
        net.train(&input, &expected);
    }

    let out = net.forward_prop(&input);

    pm(&out);
    println!("expected");
    pm(&expected);
}