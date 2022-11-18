use matrix::Matrix;
use matrix::IMatrix;
use mnist::Reader;
use net::Network;
use one_hot::OneHot;
use num::Num;

use crate::mnist::DataType;

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

    let mut network = Network::new([784, 25, 10]);

    for i in 0..1000 {
        network.train(mnist.train_images(), mnist.train_labels(), 10);
    }

    let i = 3;

    println!("{}", mnist.image_string(DataType::Train, i));

    let out = network.forward_prop(&mnist.train_images()[i]);

    println!("{}", out.to_string());

    let mut max = out[0];
    let mut index = 0;
    for (i, n) in out.iter().enumerate() {
        if n > max {
            max = n;
            index = i as i32;
        }
    }

    println!("the number is {}, {}% sure!", index, max);
    

    // network.train(mnist.train_images(), mnist.train_labels(), 5000);

    // let testing = [0, 1, 2];

    // for i in testing {  
    //     let image = &mnist.train_images()[i];
    //     let label = &mnist.train_labels()[i];
    
    //     println!("{}", mnist.image_string(DataType::Train, i));
    
    //     let out = network.forward_prop(image);
    
    //     println!("{}", out.to_string());
    
    //     let mut max = out[0];
    //     let mut index = 0;
    //     for (i, n) in out.iter().enumerate() {
    //         if n > max {
    //             max = n;
    //             index = i as i32;
    //         }
    //     }
    
    //     println!("the number is {}! (expected {})", index, label.hot());
    // }
}