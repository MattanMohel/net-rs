use std::path::Path;

use linalg::{Matrix, Vector};
use mnist::Reader;
use net::Net;

use crate::{mnist::{DataType::*, TEST_IMAGES, TRAIN_IMAGES}, linalg::{LinAlgGen, LinAlgMul}};


pub mod array;
pub mod net;
pub mod mnist;
pub mod cost;
pub mod step;
pub mod num;
pub mod linalg;

fn main() {

    // let target = [Vector::from_buf(3, vec![0.23_f32, 0.9, 0.01])];
    // let input = [Vector::from_buf(3, vec![0.30_f32, 0.20, 0.86])];

    // let mut net = net::Net::new([3, 5, 3]);

    // for _ in 0..100000 {
    //     net.train(&input, &target, 1);
    // }

    // let output = net.forward_prop(&input[0]);

    // println!(
    //     "output: \n\n{} \n\nexpected: \n\n{}",
    //     output.as_string(),
    //     target[0].as_string()
    // );

    // for i in 40..45 {  
    //     println!("{}", mnist.image_string(Train, i));
        
    //     let image = &mnist.train_images()[i];
    //     let out = network.forward_prop(image);
    
    //     println!("{}", out.as_string());
        
    //     let num = out.hot(); 
        
    //     println!("image {}: {}% - {}", i, out[num], num);
    // }
    
    let mnist = Reader::new();

    // let mut net = Net::new([784, 248, 124, 10])
    //     .with_learn_rate(0.015)
    //     .with_epoch_stats(true)
    //     .with_error_stats(true)
    //     .with_dir("src/models/digit.json")
    //     .build();

    let mut net = Net::<4>::from_file("src/models/digit.json");
    net.data
        .with_epoch_stats(false)
        .with_error_stats(false);

    for i in 900..910 {
        let prop = net.forward_prop(&mnist.test_images()[i]);
        println!("\n{}\n", mnist.image_string(Test, i));
        println!("preidcted: {} with {}%", prop.hot(), prop[prop.hot()]);
    }
    
    // net.save();
}