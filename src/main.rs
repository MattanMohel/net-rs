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
    let m1 = Matrix::<i32>::from_arr([[0, 1, 2, 3], 
                                      [4, 5, 6, 7], 
                                      [8, 9, 10, 11]]);

    let cols = m1.cols(0, 2, 2);

    println!("dim: {:?}, t: {:?}", m1.dim(), cols.dim());

    for i in cols.iter() {
        println!("[value: {i}]")
    }

    // for v in m1.cols(0, m1.row(), 1).iter() {
    //     println!("{v}");
    // }
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