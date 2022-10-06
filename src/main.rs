pub mod mat;
pub mod num;
// pub mod net;
// pub mod act;
// pub mod cost;
// pub mod init;

use mat::Mat;
use mat::MatErr;
use num::Num;

fn main() -> Result<(), MatErr> {

    let mat = Mat::<i32>::from_arr([[0, 1, 2], [3, 4, 5]]);

    for i in mat.rows() {
        println!("{:?}", i);
    }

    println!();

    let transpose = mat.transpose();

    for i in transpose.rows() {
        println!("{:?}", i);
    }
    
    
    Ok(())
}