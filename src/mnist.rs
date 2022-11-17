use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};

use super::matrix::Matrix;

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
const TRAIN_LABELS_PATH: &str = "res//train_labels";
const TEST_LABELS_PATH: &str = "res//test_labels";
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
const TRAIN_IMAGES_PATH: &str = "res//train_images";
const TEST_IMAGES_PATH: &str = "res//test_images";
const TRAIN_IMAGES: usize = 60_000;
const TEST_IMAGES:  usize = 10_000;
const IMAGE_DATA_OFFSET: usize = 16;
const BYTES_PER_IMAGE: usize = 28*28;

/// Struct for reading and storing the 
/// MNIST dataset data in a '.jpg' format
pub struct Reader {
    train_images: Vec<Matrix>,
    train_labels: Vec<Matrix>,
    
    test_images: Vec<Matrix>,
    test_labels: Vec<Matrix>
}

enum DataType {
    Train,
    Test
}

impl Reader {
    pub fn new(seed: Option<u64>, shuffle: bool) -> Self {
        let train_images = Self::read_images(DataType::Train);
        let train_labels = Self::read_labels(DataType::Train);

        let test_images = Self::read_images(DataType::Test);
        let test_labels = Self::read_labels(DataType::Test);


        todo!()
    }

    fn read_labels(data_type: DataType) -> Vec<u8> {
        let path;
        match data_type {
            DataType::Train => path = TRAIN_LABELS_PATH,
            DataType::Test =>  path = TEST_LABELS_PATH
        }

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

    fn read_images(data_type: DataType) -> Vec<u8> {
        let path;
        match data_type {
            DataType::Train => path = TRAIN_IMAGES_PATH,
            DataType::Test =>  path = TEST_IMAGES_PATH
        }

        let mut file = File::open(path).expect("couldn't open image path");
        let mut images = Vec::new(); // TODO: initiate with capacity

        file.read_to_end(&mut images).expect("couldn't read images");

        assert_eq!(Self::as_u32(&images[0..4]), IMAGE_MAGIC_NUMBER);

        images.drain(0..IMAGE_DATA_OFFSET);
        
        match data_type {
            DataType::Train => assert_eq!(images.len(), BYTES_PER_IMAGE*TRAIN_LABELS),
            DataType::Test  => assert_eq!(images.len(), BYTES_PER_IMAGE*TEST_LABELS)
        }

        images
    }

    fn parse_images(data_type: DataType, bytes: &Vec<u8>) -> Vec<Matrix> {
        let mut images;

        match data_type {
            DataType::Train => images = Vec::with_capacity(TRAIN_IMAGES),
            DataType::Test  => images = Vec::with_capacity(TEST_IMAGES),
        }
        
        for i in 0..bytes.len()/BYTES_PER_IMAGE {
            let image = bytes[i*BYTES_PER_IMAGE..(i+1)*BYTES_PER_IMAGE]
                .map(|n| n as N)
                .collect();

            images.push(Matrix::from((BYTES_PER_IMAGE, 1), image));
        }

        images
    }

    fn parse_labels(data_type: DataType, bytes: &Vec<u8>) -> Vec<Matrix> {
        let mut labels;

        let buf = bytes.iter().map(|n| n as N).collect();

        Matrix::from_buf(())

        match data_type {
            DataType::Train => images = Vec::with_capacity(TRAIN_IMAGES),
            DataType::Test  => images = Vec::with_capacity(TEST_IMAGES),
        }
        
        for i in 0..bytes.len()/BYTES_PER_IMAGE {
            let image = bytes[i*BYTES_PER_IMAGE..(i+1)*BYTES_PER_IMAGE].collect();
            images.push(Matrix::from((BYTES_PER_IMAGE, 1), image));
        }

        images
    }
    
    fn as_u32(buf: &[u8]) -> u32 {
        let bytes = [buf[0], buf[1], buf[2], buf[3]];
        u32::from_be_bytes(bytes)
    }
}