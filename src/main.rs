use std::ops::Deref;

pub mod mat;
pub mod num;
// pub mod net;
pub mod act;
pub mod cost;
pub mod mat_test;

// use net::Net;

use mat_test::*;

fn main() {
    let m1 = Matrix::<i32>::from_arr([[0, 1], [2, 3], [4, 5]]);
    // let m2 = m1.transpose();

    for i in m1.slice(m1.dim_inv(), |(i, j)| (j, i)).iter() {
        println!("{i} ")
    }
    // let mut net = Net::new([2, 3, 2])?;

    // println!("net: {net}\n\n");

    // let input = Mat::<N>::from_arr([[0.4], [0.87]]);
    // let output = Mat::<N>::from_arr([[0.0], [1.0]]);

    // let prop = net.forward_prop(&input)?;

    // println!("prop: {prop}\n\n");

    // net.train(&input, &output)?;

    // let prop = net.forward_prop(&input)?;

    // println!("prop: {prop}\n\n");

    // // for _ in 0..100 {

    // // }
}