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
    let target = [Matrix::from_buf((3, 1), vec![0.23_f32, 0.9, 0.01])];
    let input = [Matrix::from_buf((3, 1), vec![0.30_f32, 0.20, 0.86])];

    let mut net = Network::new([3, 5, 3]);

    for _ in 0..10000 {
        net.train(&input, &target, 1);
    }

    let output = net.forward_prop(&input[0]);

    println!(
        "output: \n\n{} \n\nexpected: \n\n{}",
        output.to_string(),
        target[0].to_string()
    );

    // let mnist = Reader::new();

    // let mut network = Network::new([784, 128, 64, 10]);

    // for _ in 0..100 {
    //     network.train(mnist.train_images(), mnist.train_labels(), 300);
    // }

    // let testing = [92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 300, 303, 344, 666, 767 ,5675, 7875, 56756, 6878, 544];

    // for i in testing {  
    //     let image = &mnist.train_images()[i];
    
    //     println!("{}", mnist.image_string(DataType::Train, i));
    
    //     let out = network.forward_prop(image);
    
    //     println!("{}", out.to_string());
    
    //     let mut max = out[0];
    //     let mut index = 0;
    //     for (i, n) in out.iter().enumerate() {
    //         if n > max {
    //             max = n;
    //             index = i;
    //         }
    //     }
    
    //     println!("image {}: the number is {}, {}% sure!", i, index, max);
    // }
}