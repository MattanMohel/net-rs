pub mod mat;
pub mod num;
pub mod net;
pub mod act;
pub mod cost;
pub mod init;

use num::N;
use mat::Mat;
use mat::MatErr;
use net::Net;

fn main() -> Result<(), MatErr> {
    let m1 = Mat::<f64>::from_arr([[1., 4., 6.], [5., 9., 7.], [8., 11., 13.], [99., 0., 7.7]]);
    let m2 = Mat::<f64>::from_arr([[1., 4.], [5., 9.], [8., 11.]]);

    let p1 = m1.mul(&m2)?;

    println!("{}", p1);

    Ok(())




    // let net = Net::new([5, 7, 5, 3]);

    // println!("{}", net);

    // let prop = net.forward_prop(Mat::<f64>::from_arr([[1., 2., 3., 4., 5.]]));

    // println!("prop: {}", prop);

    // Ok(())
}