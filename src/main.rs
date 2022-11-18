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

    let mut network = Network::new([784, 128, 64, 10]);

    for _ in 0..100 {
        network.train(mnist.train_images(), mnist.train_labels(), 300);
    }

    let testing = [92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 300, 303, 344, 666, 767 ,5675, 7875, 56756, 6878, 544];

    for i in testing {  
        let image = &mnist.train_images()[i];
    
        println!("{}", mnist.image_string(DataType::Train, i));
    
        let out = network.forward_prop(image);
    
        println!("{}", out.to_string());
    
        let mut max = out[0];
        let mut index = 0;
        for (i, n) in out.iter().enumerate() {
            if n > max {
                max = n;
                index = i;
            }
        }
    
        println!("image {}: the number is {}, {}% sure!", i, index, max);
    }
}