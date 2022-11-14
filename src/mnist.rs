use std::fs;
use std::path::PathBuf;
use image::DynamicImage;
use image::io::Reader as ImageReader;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};

use crate::mat::Matrix;
use crate::num::N;

const TRAIN_SAMPLES: usize = 60_000;
const TEST_SAMPLES:  usize = 40_000;
const TRAIN_ROOT_DIR: &str = "C://repo//res//MNIST//training";
const TEST_ROOT_DIR:  &str = "C://repo//res//MNIST//testing";
const SAMPLE_DIM: usize = 28*28;
const DIGITS: usize = 3;

// TODO: Speed up - this is unbearably slow
// probably needs a complete rework
// switch to READBUF

/// Sample of (expected number, image source)
pub type Sample = (Matrix, Matrix);

/// Struct for reading and storing the 
/// MNIST dataset data in a '.jpg' format
pub struct Reader {
    train_data: Vec<Sample>,
    test_data:  Vec<Sample>,
    shuffle_seed: StdRng 
}

impl Reader {
    pub fn new(seed: Option<u64>, shuffle: bool) -> Self {
        let shuffle_seed;

        match seed {
            Some(n) => shuffle_seed = StdRng::seed_from_u64(n),
            None => shuffle_seed = StdRng::from_rng(thread_rng()).unwrap()
        }

        let mut reader = Self 
        {
            train_data: Vec::with_capacity(TRAIN_SAMPLES),
            test_data:  Vec::with_capacity(TEST_SAMPLES), 
            shuffle_seed
        };

        // absoluet paths to train and test directories
        let train_path = PathBuf::from(TRAIN_ROOT_DIR);
        let test_path  = PathBuf::from(TEST_ROOT_DIR);

        for digit in 0..DIGITS {
            let digit_string   = digit.to_string();

            // concatenate into absolute path for current digit samples
            let num_train_path = train_path.join(PathBuf::from(&digit_string));
            let num_test_path  = test_path.join(PathBuf::from(&digit_string));

            // create directory iterator for all digit sampels
            let train_dir_iter = fs::read_dir(num_train_path).unwrap();
            let test_dir_iter  = fs::read_dir(num_test_path).unwrap();

            // collect digit train samples
            let mut train_data = train_dir_iter
                .map(|sample| {
                    let path = sample.unwrap().path();
                    let image = ImageReader::open(path).unwrap().decode().unwrap();

                    let mut digit_matrix = Matrix::zeros((DIGITS, 1));
                    digit_matrix[(digit, 0)] = 1.0 as N;

                    let buf = image.as_bytes().iter().map(|n| (*n as N) / 255.0 as N).collect();
                    (digit_matrix, Matrix::from_buf((SAMPLE_DIM, 1), buf))     
                })
                .collect();
                
            // collect digit test samples
            let mut test_data = test_dir_iter
                .map(|sample| {
                    let path = sample.unwrap().path();
                    let image = ImageReader::open(path).unwrap().decode().unwrap();

                    let mut digit_matrix = Matrix::zeros((DIGITS, 1));
                    digit_matrix[(digit, 0)] = 1.0 as N;

                    let buf = image.as_bytes().iter().map(|n| (*n as N) / 255.0 as N).collect();
                    (digit_matrix, Matrix::from_buf((SAMPLE_DIM, 1), buf))             
                })
                .collect();
            
            // append digit data
            reader.train_data.append(&mut train_data);
            reader.test_data.append(&mut test_data);

            println!("read {digit}");
        }

        if shuffle {
            reader.shuffle_train_data();
            reader.shuffle_test_data();
        }

        reader
    }

    pub fn set_seed(&mut self, seed: Option<u64>) {
        match seed {
            Some(n) => self.shuffle_seed = StdRng::seed_from_u64(n),
            None => self.shuffle_seed = StdRng::from_rng(thread_rng()).unwrap()
        }
    }

    pub fn shuffle_train_data(&mut self) {
        self.train_data.shuffle(&mut self.shuffle_seed);
    }

    pub fn shuffle_test_data(&mut self) {
        self.test_data.shuffle(&mut self.shuffle_seed);
    }
}