use linalg::{Matrix, Vector};
use mnist::Reader;
use net::Network;

use crate::{mnist::DataType::*, linalg::{LinAlgGen, LinAlgMul}};


pub mod array;
pub mod net;
pub mod mnist;
pub mod cost;
pub mod step;
pub mod num;
pub mod linalg;

fn main() {
    let m1 = Matrix::from_buf((2, 3), vec![3., 6.1, 4.2, 1., 5., 8.]);
    let m2 = Vector::from_buf(3, vec![1., 2.1, 3.4]);

    let prod: Matrix = m1.mul(&m2);

    println!("prod: \n\n{}", prod.to_string());
    let mnist = Reader::new();

    let mut network = Network::new([784, 50, 10]);

    for i in 0..1000 {
        network.train(mnist.train_images(), mnist.train_labels(), 10);
        println!("epoch {}/100", i);
    }

    let testing = [0];

    for i in testing {  
        let image = &mnist.train_images()[i];
    
        println!("{}", mnist.image_string(Train, i));
    
        let out = network.forward_prop(image);
    
        println!("{}", out.to_string());
    
        let mut max = out[0];
        let mut index = 0;
        for (i, n) in out.buf().iter().enumerate() {
            if *n > max {
                max = *n;
                index = i;
            }
        }
    
        println!("image {}: the number is {}, {}% sure!", i, index, max);
    }
}