use linalg::{Matrix, Vector};
use mnist::Reader;
use net::Net;

use crate::{mnist::DataType::*, linalg::{LinAlgGen, LinAlgMul}};


pub mod array;
pub mod net;
pub mod mnist;
pub mod cost;
pub mod step;
pub mod num;
pub mod linalg;
pub mod network;

fn main() {
    
    // let m1 = Matrix::<i32>::from_buf((4, 3), vec![1, 2, 3, 4, 5, 6, 23, 4, 11, 1, 98, 12]);
    // let m2 = Matrix::<i32>::from_buf((2, 3), vec![7, 8, 9, 1, 2, 3]);

    // let p1 = m1.mul::<_, Matrix::<i32>>(&m2.transpose());

    // println!("p1: {}", p1.to_string());

    let target = [Matrix::from_buf((3, 1), vec![0.23_f32, 0.9, 0.01])];
    let input = [Matrix::from_buf((3, 1), vec![0.30_f32, 0.20, 0.86])];

    let mut net = network::Network::new([3, 5, 3]);

    for _ in 0..100000 {
        net.train(&input, &target, 1);
    }

    let output = net.forward_prop(&input[0]);

    println!(
        "output: \n\n{} \n\nexpected: \n\n{}",
        output.as_string(),
        target[0].as_string()
    );

    // let mnist = Reader::new();

    // let mut network = Network::new([784, 50, 10]);

    // for i in 0..1000 {
    //     network.train(mnist.train_images(), mnist.train_labels(), 10);
    //     println!("epoch {}/100", i);
    // }

    // let testing = [0];

    // for i in testing {  
    //     let image = &mnist.train_images()[i];
    
    //     println!("{}", mnist.image_string(Train, i));
    
    //     let out = network.forward_prop(image);
    
    //     println!("{}", out.to_string());
    
    //     let mut max = out[0];
    //     let mut index = 0;
    //     for (i, n) in out.buf().iter().enumerate() {
    //         if *n > max {
    //             max = *n;
    //             index = i;
    //         }
    //     }
    
    //     println!("image {}: the number is {}, {}% sure!", i, index, max);
    // }
}