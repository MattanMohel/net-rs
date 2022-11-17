use matrix::Matrix;
use matrix::IMatrix;
use mnist::Reader;
use net::Network;
use one_hot::OneHot;
use num::Num;

pub mod matrix;
pub mod num;
pub mod net;
pub mod cost;
pub mod num_fn;
pub mod activation;
pub mod mnist;
pub mod one_hot;
pub mod matrix_slice;

fn main() {
    let mnist = Reader::new();

    println!("images: {}", mnist.train_images().len());

    let _network = Network::new([3, 5, 3]);

    // let m1 = Matrix::<i32>::from_arr([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
    // let m2 = m1.transpose();
    // let prod = m1.mul(&m2);
    
    // println!("m1: {}", m1.to_string());

    // println!("det of m1: {}", prod.determinant());

    // let hot = OneHot::<i32>::new(5, 1);

    // println!("{}", hot.to_string());
}