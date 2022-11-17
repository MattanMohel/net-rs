use std::fs::{self, File};
use std::io::{Read, self};
use std::path::{PathBuf, Path};
use std::slice::Chunks;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};
use super::matrix::IMatrix;

use crate::num::N;
use crate::one_hot::OneHot;

use super::matrix::Matrix;

// PRONOT TODO: credit JonathanWoollett - download unzipped and port to resources

/// MNIST Data Formatting Information
///
/// LABEL FORMAT:
/// 
/// [offset]   [value]          [description]
/// 0000       2049             magic number
/// 0004       ???              # labels
/// 0008       ???              label
/// 0009       ???              label
/// ...
/// xxxx       ???              label

const LABEL_MAGIC_NUMBER: u32 = 2049;
const TRAIN_LABELS_PATH: &str = "src/res/train-labels-idx1-ubyte";
const TEST_LABELS_PATH: &str =  "src/res/t10k-labels-idx1-ubyte";
const TRAIN_LABELS: usize = 60_000;
const TEST_LABELS:  usize = 10_000;
const LABEL_DATA_OFFSET: usize = 8;

/// IMAGE FORMAT:
/// 
/// [offset]   [value]          [description]
/// 0000       2051             magic number
/// 0004       ???              number of images
/// 0008       28               # rows
/// 0012       28               # columns
/// 0016       ??               pixel
/// 0017       ??               pixel
/// ...
/// xxxx       ??               pixel
/// 
/// TRAIN IMAGES: 60,000
/// TEST IMAGES:  10,000

const IMAGE_MAGIC_NUMBER: u32 = 2051;
const TRAIN_IMAGES_PATH: &str = "src/res/train-images";
const TEST_IMAGES_PATH: &str =  "src/res/t10k-images-idx3-ubyte";
const TRAIN_IMAGES: usize = 60_000;
const TEST_IMAGES:  usize = 10_000;
const IMAGE_DATA_OFFSET: usize = 16;
const BYTES_PER_IMAGE: usize = 28*28;
const BYTES_PER_AXIS: usize = 28;

/// Struct for reading and storing the 
/// MNIST dataset data in a '.jpg' format
pub struct Reader {
    train_images: Vec<Matrix>,
    train_labels: Vec<OneHot>,   
    test_images: Vec<Matrix>,
    test_labels: Vec<OneHot>
}

pub enum DataType {
    Train,
    Test
}

impl Reader {
    pub fn new() -> Self {
        let env = std::env::current_dir().expect("invalid working dir");
        
        Self { 
            train_images: Self::parse_images(DataType::Train, &env),
            train_labels: Self::parse_labels(DataType::Train, &env),
            test_images:  Self::parse_images(DataType::Test, &env),
            test_labels:  Self::parse_labels(DataType::Test, &env)
        }
    }

    pub fn train_images(&self) -> &Vec<Matrix> {
        &self.train_images
    }

    pub fn train_labels(&self) -> &Vec<OneHot> {
        &self.train_labels
    }

    pub fn test_images(&self) -> &Vec<Matrix> {
        &self.test_images
    }

    pub fn test_labels(&self) -> &Vec<OneHot> {
        &self.test_labels
    }

    fn parse_images(data_type: DataType, env: &PathBuf) -> Vec<Matrix> {
        let image_bytes = Self::read_images(data_type, env);
        Self::parse_image_bytes(&image_bytes)
    }

    fn parse_labels(data_type: DataType, env: &PathBuf) -> Vec<OneHot> {
        let label_bytes = Self::read_labels(data_type, env);
        Self::parse_label_bytes(&label_bytes)
    }

    fn read_labels(data_type: DataType, env: &PathBuf) -> Vec<u8> {
        let res_path;
        match data_type {
            DataType::Train => res_path = TRAIN_LABELS_PATH,
            DataType::Test =>  res_path = TEST_LABELS_PATH
        }

        let path = env.join(res_path);
        
        let mut file = File::open(path).expect("couldn't open label path");
        let mut labels = Vec::new(); // TODO: initiate with capacity

        file.read_to_end(&mut labels).expect("couldn't read labels");

        assert_eq!(Self::as_u32(&labels[0..4]), LABEL_MAGIC_NUMBER);

        labels.drain(0..LABEL_DATA_OFFSET);
        
        match data_type {
            DataType::Train => assert_eq!(labels.len(), TRAIN_LABELS),
            DataType::Test  => assert_eq!(labels.len(), TEST_LABELS)
        }
        
        labels
    }

    fn read_images(data_type: DataType, env: &PathBuf) -> Vec<u8> {
        let res_path;
        match data_type {
            DataType::Train => res_path = TRAIN_IMAGES_PATH,
            DataType::Test =>  res_path = TEST_IMAGES_PATH
        }

        let path = env.join(res_path);

        let mut file = File::open(path).expect("couldn't open image path");
        let mut images = Vec::new(); // TODO: initiate with capacity

        file.read_to_end(&mut images).expect("couldn't read images");

        //assert_eq!(Self::as_u32(&images[0..4]), IMAGE_MAGIC_NUMBER);

        images.drain(0..IMAGE_DATA_OFFSET);
        
        //match data_type {
        //    DataType::Train => assert_eq!(images.len(), BYTES_PER_IMAGE*TRAIN_LABELS),
        //    DataType::Test  => assert_eq!(images.len(), BYTES_PER_IMAGE*TEST_LABELS)
        //}

        images
    }

    fn parse_image_bytes(bytes: &Vec<u8>) -> Vec<Matrix> {
        let images: Vec<Matrix> = bytes
            .chunks(BYTES_PER_IMAGE)
            .map(|image| {
                let buf = image
                    .iter()
                    .map(|n| *n as N)
                    .collect();

                Matrix::from_buf((BYTES_PER_IMAGE, 1), buf)
            })
            .collect();

        println!("len: {}", images.len());

        for (i, n) in images[1182].iter().enumerate() {
            if n > 150.0 {
                print!("@");
            } 
            else {
                print!(".");
            }

            if i % BYTES_PER_AXIS == 0 {
                println!();
            }
        }    

        images
    }

    fn parse_label_bytes(bytes: &Vec<u8>) -> Vec<OneHot> {
        bytes
            .iter()
            .map(|label| OneHot::new(*label as usize, 10))
            .collect()
    }
    
    fn as_u32(buf: &[u8]) -> u32 {
        let bytes = [buf[0], buf[1], buf[2], buf[3]];
        u32::from_be_bytes(bytes)
    }

    pub fn print_image(&self, data_type: DataType, index: usize) {
        let images;
        
        match data_type {
            DataType::Train => images = &self.train_images,
            DataType::Test  => images = &self.test_images
        }

        for (i, n) in images[index].iter().enumerate() {
            if n > 150.0 {
                print!("@");
            } 
            else {
                print!(".");
            }

            if i % BYTES_PER_AXIS == 0 {
                println!();
            }
        }                        
    }
}