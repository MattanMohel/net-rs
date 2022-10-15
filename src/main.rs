pub mod mat;
pub mod num;
pub mod net;
pub mod act;
pub mod cost;

use std::iter::Copied;
use std::slice::{
    Iter,
    IterMut
};
use mat::MatErrType;
use num::N;
use mat::Mat;
use mat::MatErr;
use mat::MatCollect;
use net::Net;

fn main() -> Result<(), MatErrType> {
    let m1 = Mat::<f64>::from_arr([[1., 4., 6.], [5., 9., 7.], [8., 11., 13.], [99., 0., 7.7]]);
    let m2 = Mat::<f64>::from_arr([[1., 4.], [5., 9.], [8., 11.]]);

    let i = m1.iter().map(|e| 10. * e).to_matrix((4, 3));

    // println!("{m1}");

    // let p1 = m1.mul(&m2)?;

    // println!("{p1}");

    // let mut net = Net::new([2, 3, 2]).unwrap();

    // let input = Mat::from_arr([[0.0], [0.5]]);

    // let prop = net.forward_prop(&input).unwrap();
    
    // println!("net: {net}, prop: {prop}\n\n\n");

    // let expected = Mat::from_arr([[0.], [1.]]);

    // net.train(input.clone(), expected)?;

    // let prop = net.forward_prop(&input).unwrap();

    // println!("net: {net}, prop: {prop}");

    Ok(())
}