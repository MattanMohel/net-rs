use std::fs::{self, File};
use std::io::{Read, self, BufReader};
use std::path::{PathBuf, Path};
use std::slice::Chunks;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
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
const TRAIN_LABELS_PATH: &str = "src/res/train-labels";
const TEST_LABELS_PATH: &str =  "src/res/test_labels";
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
const TEST_IMAGES_PATH: &str =  "src/res/test-images";
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
        let mut work_dir = std::env::current_dir().expect("invalid working dir");

        if work_dir.ends_with("src") {
            work_dir.pop();
        }

        Self { 
            train_images: Self::read_images(DataType::Train, &work_dir),
            train_labels: Self::read_labels(DataType::Train, &work_dir),
            test_images:  Self::read_images(DataType::Test, &work_dir),
            test_labels:  Self::read_labels(DataType::Test, &work_dir)
        }
    }

    pub fn statistics(&self) {
        println!("train images: {}", self.train_images.len());
        println!("train labels: {}", self.train_labels.len());
        println!("test images: {}", self.test_images.len());
        println!("test labels: {}", self.test_images.len());
    }

    fn read_labels(data_type: DataType, work_dir: &PathBuf) -> Vec<OneHot> {
        let path = match data_type {
            DataType::Train => TRAIN_LABELS_PATH,
            DataType::Test =>  TEST_LABELS_PATH
        };

        let mut label_bytes = std::fs::read(work_dir.join(path))
            .expect("couldnt read labels");

        assert_eq!(Self::read_be_u32(&label_bytes[0..4]), LABEL_MAGIC_NUMBER);

        label_bytes.drain(0..LABEL_DATA_OFFSET);
        
        match data_type {
            DataType::Train => assert_eq!(label_bytes.len(), TRAIN_LABELS),
            DataType::Test  => assert_eq!(label_bytes.len(), TEST_LABELS)
        }
        
        label_bytes
            .iter()
            .map(|label| OneHot::new(10, *label as usize))
            .collect()
    }

    fn read_images(data_type: DataType, work_dir: &PathBuf) -> Vec<Matrix> {
        let path = match data_type {
            DataType::Train => TRAIN_IMAGES_PATH,
            DataType::Test =>  TEST_IMAGES_PATH
        };

        
        let mut image_bytes = std::fs::read(work_dir.join(path))
            .expect("couldn't read images");
        
        assert_eq!(Self::read_be_u32(&image_bytes[0..4]), IMAGE_MAGIC_NUMBER);
        
        image_bytes.drain(0..IMAGE_DATA_OFFSET);
        
        match data_type {
           DataType::Train => assert_eq!(image_bytes.len(), BYTES_PER_IMAGE*TRAIN_LABELS),
           DataType::Test  => assert_eq!(image_bytes.len(), BYTES_PER_IMAGE*TEST_LABELS)
        }

        image_bytes
            .chunks(BYTES_PER_IMAGE)
            .map(|image| {
                let buf = image
                    .iter()
                    .map(|n| *n as N)
                    .collect();

                Matrix::from_buf((BYTES_PER_IMAGE, 1), buf)
            })
            .collect() 
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

    pub fn image_string(&self, data_type: DataType, index: usize) -> String {
        let images;
        
        match data_type {
            DataType::Train => images = &self.train_images,
            DataType::Test  => images = &self.test_images
        }

        images[index]
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let ch = match n as u8 {
                    0 => " ",
                    0..=50   => "+",
                    51..=100 => "#",
                    _ => "@",
                };

                if i % BYTES_PER_AXIS == 0 {
                    format!(" {}\n", ch)
                }
                else {
                    format!(" {}", ch)
                }
            })
            .collect()
    }

    fn read_be_u32(buf: &[u8]) -> u32 {
        let buf = [buf[0], buf[1], buf[2], buf[3]];
        u32::from_be_bytes(buf)
    }
}