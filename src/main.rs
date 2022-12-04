use std::path::Path;

use draw::run_sketch;
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
mod draw;

fn main() {   
    let mnist = Reader::new();

    // let mut net = Net::new([784, 248, 124, 10])
    //     .with_learn_rate(0.015)
    //     .with_epoch_stats(true)
    //     .with_error_stats(true)
    //     .with_dir("src/models/digit.json")
    //     .build();

    // net.train();
    // net.save();

    let mut net = Net::<4>::from_file("src/models/digit_hp.json");

    let acc = net.accuracy(&mnist.test_images(), &mnist.test_labels());
    println!("acc: {}", acc);
    // net.data
    //     .with_epoch_stats(false)
    //     .with_error_stats(false);

    // for i in 345..355 {
    //     let prop = net.forward_prop(&mnist.test_images()[i]);
    //     println!("\n{}\n", mnist.image_string(Test, i));
    //     println!("preidcted: {} with {}%", prop.hot(), prop[prop.hot()]);
    // } 

    run_sketch();
}