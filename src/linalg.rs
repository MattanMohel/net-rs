use std::ops::{Index, IndexMut};

use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::num::Num;

pub trait LinAlg<N: Num=f32> 
where
    Self: LinAlgGen<N> + LinAlgMul<N> + Clone
{}

pub trait LinAlgGen<N: Num=f32> 
where 
    Self: Sized + Index<Self::Dim, Output=N> + IndexMut<Self::Dim, Output=N>
{
    type Dim: Copy;

    /// Returns #rows
    fn row(&self) -> usize;

    /// Returns #columns
    fn col(&self) -> usize;

    /// Returns the #elements in given dimensions
    fn to_size(dim: Self::Dim) -> usize;

    /// Returns converted 2d dimensions to Dim
    fn to_dim(rc: (usize, usize)) -> Self::Dim;

    /// Returns converted Dim to 2d dimensions
    fn to_shape(dim: Self::Dim) -> (usize, usize);

    /// Returns reference to buffer
    fn buf(&self) -> &Vec<N>;

    /// Returns mutable reference to buffer
    fn buf_mut(&mut self) -> &mut Vec<N>;

    /// Creates matrix from buffer and dimensions
    fn from_buf(dim: Self::Dim, buf: Vec<N>) -> Self;

    /// Returns the dimensions
    fn shape(&self) -> (usize, usize) {
        (self.row(), self.col())
    }

    /// Creates matrix of filled with zeros
    fn from_zeros(dim: Self::Dim) -> Self {
        Self::from_buf(dim, vec![N::zero(); Self::to_size(dim)])
    } 

    /// Create matrix filled with a value
    fn from_fill(dim: Self::Dim, value: N) -> Self {
        Self::from_buf(dim, vec![value; Self::to_size(dim)])
    } 

    /// Create matrix mapped by index
    fn from_map<F: FnMut(Self::Dim) -> N>(dim: Self::Dim, mut map: F) -> Self {
        let (_, col) = Self::to_shape(dim);

        let buf = (0..Self::to_size(dim))
            .map(|i| {
                let rc = (i / col, i % col);
                map(Self::to_dim(rc))
            })
            .collect();

        Self::from_buf(dim, buf)
    }

    /// Returns a new matrix with random values between [-1, 1]
    fn rand(dim: Self::Dim) -> Self 
    where 
        Standard: Distribution<N>
    {
        let mut rng = rand::thread_rng();
        Self::from_map(dim, |_| rng.gen_range(-N::one()..N::one()))
    }

    /// Returns the matrix transpose
    fn transpose(&self) -> Matrix<N> {
        let mut buf = Vec::with_capacity(self.row() * self.col());

        for c in 0..self.col() {
            for r in 0..self.row() {
                buf.push(self.buf()[self.col() * r + c]);
            }
        }

        Matrix::from_buf((self.col(), self.row()), buf)
    }

    /// Sets self to element-wise multipication matrix
    fn dot_eq<M: LinAlgGen<N>>(&mut self, rhs: &M) -> &mut Self {
        if self.shape() != rhs.shape() {
            panic!("unmatched 'dot-eq' dimensions {:?} | {:?}", self.shape(), rhs.shape())
        }

        for (n, other) in self.buf_mut().iter_mut().zip(rhs.buf().iter()) {
            *n *= *other;
        }
        
        self
    }

    /// Returns a new element-wise multipication matrix
    fn dot<M: LinAlgGen<N>>(&self, rhs: &M) -> Self {
        if self.shape() != rhs.shape() {
            panic!("unmatched 'dot' dimensions {:?} | {:?}", self.shape(), rhs.shape())
        }

        let buf = self
            .buf()
            .iter()
            .zip(rhs.buf().iter()) 
            .map(|(n, other)| *n * *other)
            .collect();

        Self::from_buf(Self::to_dim(self.shape()), buf)
    }

    /// Returns matrix scaled by scalar
    fn scale(&self, scalar: N) -> Self {
        let buf = self
            .buf()
            .iter()
            .map(|n| *n * scalar)
            .collect();
        
        Self::from_buf(Self::to_dim(self.shape()), buf)
    }

    /// Scales self by scalar
    fn scale_eq(&mut self, scalar: N) -> &mut Self {
        for n in self.buf_mut().iter_mut() {
            *n *= scalar;
        }

        self
    }

    /// Sets all elements of self to value
    fn fill_eq(&mut self, value: N) -> &mut Self {
        for n in self.buf_mut().iter_mut() {
            *n = value;
        }

        self
    }

    /// Creates matrix from element map
    fn map<F: Fn(N) -> N>(&self, map: F) -> Self {
        let buf = self
            .buf()
            .iter()
            .map(|n| map(*n))
            .collect();
        
        Self::from_buf(Self::to_dim(self.shape()), buf)
    }

    /// Maps all elements of self by map
    fn map_eq<F: Fn(N) -> N>(&mut self, map: F) -> &mut Self {
        for n in self.buf_mut().iter_mut() {
            *n = map(*n);
        }

        self
    }

    /// Sets self to the sum of the two matrices
    fn add_eq<M: LinAlgGen<N>>(&mut self, rhs: &M) -> &mut Self {
        if self.shape() != rhs.shape() {
            panic!("unmatched 'add-eq' dimensions {:?} | {:?}", self.shape(), rhs.shape())
        }

        for (n, other) in self.buf_mut().iter_mut().zip(rhs.buf().iter()) {
            *n += *other;
        }
        
        self
    }

    /// Returns new sum matrix
    fn add<M: LinAlgGen<N>>(&self, rhs: &M) -> Self {
        if self.shape() != rhs.shape() {
            panic!("unmatched 'add' dimensions {:?} | {:?}", self.shape(), rhs.shape())
        }

        let buf = self
            .buf()
            .iter()
            .zip(rhs.buf().iter())
            .map(|(n, other)| *n + *other)
            .collect();
        
        Self::from_buf(Self::to_dim(self.shape()), buf)
    }

    /// Sets self to the difference of the two matrices
    fn sub_eq<M: LinAlgGen<N>>(&mut self, rhs: &M) -> &mut Self {
        if self.shape() != rhs.shape() {
            panic!("unmatched 'sub-eq' dimensions {:?} | {:?}", self.shape(), rhs.shape())
        }

        for (n, other) in self.buf_mut().iter_mut().zip(rhs.buf().iter()) {
            *n -= *other;
        }

        self
    }

    /// Returns new difference matrix
    fn sub<M: LinAlgGen<N>>(&self, rhs: &M) -> Self {
        if self.shape() != rhs.shape() {
            panic!("unmatched 'sub' dimensions {:?} | {:?}", self.shape(), rhs.shape())
        }

        let buf = self
            .buf()
            .iter()
            .zip(rhs.buf().iter())
            .map(|(n, other)| *n - *other)
            .collect();
        
        Self::from_buf(Self::to_dim(self.shape()), buf)
    }

    fn to_string(&self) -> String {
        let len = 2*self.row()*self.col() + self.row();
        let mut buf = String::with_capacity(len);

        for r in 0..self.row() {
            for c in 0..self.col() {
                buf.push_str(format!("{} ", self[Self::to_dim((r, c))]).as_str());
            }
            buf.push('\n');
        } 

        buf
    }
}

pub trait LinAlgMul<N: Num=f32> 
where
    Self: LinAlgGen<N>,
{
    fn mul<M, K>(&self, rhs: &M) -> K 
    where
        M: LinAlg<N>, 
        K: LinAlg<N>
    {
        let mut buf = K::from_zeros(K::to_dim((self.row(), rhs.col())));
        Self::mul_to(self, rhs, &mut buf);
        buf
    }

    fn mul_t1<M, K>(&self, rhs: &M) -> K 
    where
        M: LinAlg<N>, 
        K: LinAlg<N>
    {
        let mut buf = K::from_zeros(K::to_dim((self.col(), rhs.col())));
        Self::mul_t1_to(self, rhs, &mut buf);
        buf
    }

    fn mul_t2<M, K>(&self, rhs: &M) -> K 
    where
        M: LinAlg<N>, 
        K: LinAlg<N>
    {
        let mut buf = K::from_zeros(K::to_dim((self.row(), rhs.row())));
        Self::mul_t2_to(self, rhs, &mut buf);
        buf
    }
    
    /// Multiplies lhs and rhs, using self as a buffer
    fn mul_to<'a, M, K>(&self, rhs: &M, buf: &'a mut K) -> &'a mut K
    where
        M: LinAlg<N>,
        K: LinAlg<N>
    {
        if self.col() != rhs.row() {
            panic!("DEDAULT IMPL - unmatched 'mul' dimensions {:?} | {:?}", self.shape(), rhs.shape())
        }

        for i in 0..self.row() {
            for j in 0..rhs.col() {
                let mut acc = N::zero();
                for k in 0..self.col() {
                    acc += self.buf()[i * self.col() + k] * rhs.buf()[k * rhs.col() + j];
                }

                buf.buf_mut()[rhs.col() * i + j] = acc;
            }
        }

        buf
    }

    /// Multiplies transpose lhs and rhs, using self as a buffer
    fn mul_t1_to<'a, M, K>(&self, rhs: &M, buf: &'a mut K) -> &'a mut K
    where
        M: LinAlg<N>,
        K: LinAlg<N>
    {
        unimplemented!("no default mul_t1_to impl")
    }

    /// Multiplies lhs and transpose rhs, using self as a buffer
    fn mul_t2_to<'a, M, K>(&self, rhs: &M, buf: &'a mut K) -> &'a mut K
    where
        M: LinAlg<N>,
        K: LinAlg<N>
    {
        unimplemented!("no default mul_t2_to impl")  
    }
}

impl<T: LinAlgGen<i32>> LinAlgMul<i32> for T {}
impl<T: LinAlgGen<i64>> LinAlgMul<i64> for T {}


impl<T: LinAlgGen<f32>> LinAlgMul<f32> for T {
    
    // fn mul_to<'a, M, K>(&self, rhs: &M, buf: &'a mut K) -> &'a mut K
    // where
    //     M: LinAlg<f32>,
    //     K: LinAlg<f32>
    // {
    //     if self.col() != rhs.row() {
    //         panic!("unmatched 'mul' dimensions {:?} | {:?}", self.shape(), rhs.shape())
    //     }

    //     unsafe {
    //         matrixmultiply::sgemm(
    //             self.row(), // m dimension
    //             self.col(), // k dimension
    //             rhs.col(),  // n dimension
    //             1_f32,
    //             self.buf().as_ptr(), // m x k matrix
    //             self.col() as isize, // row stride
    //             1,                   // col stride
    //             rhs.buf().as_ptr(),  // k x n matrix
    //             rhs.col() as isize,  // row stride
    //             1,                   // col stride
    //             1_f32,
    //             buf.buf_mut().as_mut_ptr(), // m x n buffer 
    //             buf.col() as isize,         // row stride
    //             1                           // col stride
    //         );
    //     }

    //     buf
    // }

    // fn mul_t1_to<'a, M, K>(&self, rhs: &M, buf: &'a mut K) -> &'a mut K
    // where
    //     M: LinAlg<f32>,
    //     K: LinAlg<f32>
    // { 
    //     if self.row() != rhs.row() {
    //         panic!("unmatched 'mul_t1' dimensions {:?} | {:?}", self.shape(), rhs.shape())
    //     }

    //     unsafe {
    //         matrixmultiply::sgemm(
    //             self.col(), // m dimension
    //             self.row(), // k dimension
    //             rhs.col(),  // n dimension
    //             1_f32,
    //             self.buf().as_ptr(), // m x k matrix
    //             1,                   // row stride
    //             self.col() as isize, // col stride
    //             rhs.buf().as_ptr(),  // k x n matrix
    //             rhs.col() as isize,  // row stride
    //             1,                   // col stride
    //             1_f32,
    //             buf.buf_mut().as_mut_ptr(), // m x n buffer 
    //             buf.col() as isize,         // row stride
    //             1                           // col stride
    //         );
    //     }

    //     buf
    // }

    // fn mul_t2_to<'a, M, K>(&self, rhs: &M, buf: &'a mut K) -> &'a mut K
    // where
    //     M: LinAlg<f32>,
    //     K: LinAlg<f32>
    // { 
    //     if self.col() != rhs.col() {
    //         panic!("unmatched 'mul_t2' dimensions {:?} | {:?}", self.shape(), rhs.shape())
    //     }

    //     unsafe {
    //         matrixmultiply::sgemm(
    //             self.row(), // m dimension
    //             self.col(), // k dimension
    //             rhs.row(),  // n dimension
    //             1_f32,
    //             self.buf().as_ptr(), // m x k matrix
    //             self.col() as isize, // row stride
    //             1,                   // col stride
    //             rhs.buf().as_ptr(),  // k x n matrix
    //             1,                   // row stride
    //             rhs.col() as isize,  // col stride
    //             1_f32,
    //             buf.buf_mut().as_mut_ptr(), // m x n buffer 
    //             buf.col() as isize,         // row stride
    //             1                           // col stride
    //         );
    //     }

    //     buf
    // }
}

impl<T: LinAlgGen<f64>> LinAlgMul<f64> for T {
    
    // fn mul_to<'a, M, K>(&self, rhs: &M, buf: &'a mut K) -> &'a mut K 
    // where
    //     M: LinAlg<f64>,
    //     K: LinAlg<f64>
    // {  
    //     if self.col() != rhs.row() {
    //         panic!("unmatched 'mul' dimensions {:?} | {:?}", self.shape(), rhs.shape())
    //     }

    //     unsafe {
    //         matrixmultiply::dgemm(
    //             self.row(), // m dimension
    //             self.col(), // k dimension
    //             rhs.col(),  // n dimension
    //             1_f64,
    //             self.buf().as_ptr(), // m x k matrix
    //             self.col() as isize, // row stride
    //             1,                   // col stride
    //             rhs.buf().as_ptr(),  // k x n matrix
    //             rhs.col() as isize,  // row stride
    //             1,                   // col stride
    //             1_f64,
    //             buf.buf_mut().as_mut_ptr(), // m x n buffer 
    //             buf.col() as isize,         // row stride
    //             1                           // col stride
    //         );
    //     }

    //     buf
    // }

    // fn mul_t1_to<'a, M, K>(&self, rhs: &M, buf: &'a mut K) -> &'a mut K 
    // where
    //     M: LinAlg<f64>,
    //     K: LinAlg<f64>
    // {    
    //     if self.row() != rhs.row() {
    //         panic!("unmatched 'mul_t1' dimensions {:?} | {:?}", self.shape(), rhs.shape())
    //     }

    //     unsafe {
    //         matrixmultiply::dgemm(
    //             self.col(), // m dimension
    //             self.row(), // k dimension
    //             rhs.col(),  // n dimension
    //             1_f64,
    //             self.buf().as_ptr(), // m x k matrix
    //             1,                   // row stride
    //             self.col() as isize, // col stride
    //             rhs.buf().as_ptr(),  // k x n matrix
    //             rhs.col() as isize,  // row stride
    //             1,                   // col stride
    //             1_f64,
    //             buf.buf_mut().as_mut_ptr(), // m x n buffer 
    //             buf.col() as isize,         // row stride
    //             1                           // col stride
    //         );
    //     }

    //     buf
    // }

    // fn mul_t2_to<'a, M, K>(&self, rhs: &M, buf: &'a mut K) -> &'a mut K 
    // where
    //     M: LinAlg<f64>,
    //     K: LinAlg<f64>
    // {   
    //     if self.col() != rhs.col() {
    //         panic!("unmatched 'mul_t2' dimensions {:?} | {:?}", self.shape(), rhs.shape())
    //     }

    //     unsafe {
    //         matrixmultiply::dgemm(
    //             self.row(), // m dimension
    //             self.col(), // k dimension
    //             rhs.row(),  // n dimension
    //             1_f64,
    //             self.buf().as_ptr(), // m x k matrix
    //             self.col() as isize, // row stride
    //             1,                   // col stride
    //             rhs.buf().as_ptr(),  // k x n matrix
    //             1,                   // rpw stride
    //             rhs.col() as isize,  // cpl stride
    //             1_f64,
    //             buf.buf_mut().as_mut_ptr(), // m x n buffer 
    //             buf.col() as isize,         // row stride
    //             1                           // col stride
    //         );
    //     }

    //     buf
    // }
}

#[derive(Clone)]
pub struct Matrix<N: Num=f32> {
    buf: Vec<N>,
    row: usize,
    col: usize
}

impl<N: Num> LinAlg<N> for Matrix<N> where Matrix<N>: LinAlgMul<N> 
{}

impl<N: Num> Index<(usize, usize)> for Matrix<N> {
    type Output=N;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.buf[row*self.col + col]
    }
}

impl<N: Num> IndexMut<(usize, usize)> for Matrix<N> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.buf[row*self.col + col]
    }
}

impl<N: Num> LinAlgGen<N> for Matrix<N> {
    type Dim = (usize, usize);

    fn row(&self) -> usize {
        self.row
    }

    fn col(&self) -> usize {
        self.col
    }

    fn to_size((row, col): Self::Dim) -> usize {
        row * col
    }
 
    fn to_dim(rc: (usize, usize)) -> Self::Dim {
        rc
    }

    fn to_shape(dim: Self::Dim) -> (usize, usize) {
        dim
    }
    
    fn buf(&self) -> &Vec<N> {
        &self.buf
    }

    fn buf_mut(&mut self) -> &mut Vec<N> {
        &mut self.buf
    }

    fn from_buf((row, col): Self::Dim, buf: Vec<N>) -> Self {
        Self {
            buf,
            row,
            col
        }
    }
}

impl<N: Num> Matrix<N> {
    pub fn from_arr<const R: usize, const C: usize>(arr: [[N; C]; R]) -> Self {
        let buf = arr
            .iter()
            .copied()
            .flatten()
            .collect();

        Self {
            buf,
            row: R,
            col: C
        }
    }

    pub fn to_vector(&self) -> Vector<N> {
        if self.col() != 1 {
            panic!()
        }

        Vector::from_buf(self.row, self.buf.clone())
    }
}

#[derive(Clone)]
pub struct Vector<N: Num=f32> {
    buf: Vec<N>,
    row: usize
}

impl<N: Num> Index<usize> for Vector<N> {
    type Output=N;

    fn index(&self, row: usize) -> &Self::Output {
        &self.buf[row]
    }
}

impl<N: Num> IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        &mut self.buf[row]
    }
}

impl<N: Num> LinAlgGen<N> for Vector<N> {
    type Dim = usize;

    fn row(&self) -> usize {
        self.row
    }

    fn col(&self) -> usize {
        1
    }

    fn to_size(row: Self::Dim) -> usize {
        row
    }

    fn to_dim((row, _): (usize, usize)) -> Self::Dim {
        row
    }

    fn to_shape(dim: Self::Dim) -> (usize, usize) {
        (dim, 1)
    }
    
    fn buf(&self) -> &Vec<N> {
        &self.buf
    }

    fn buf_mut(&mut self) -> &mut Vec<N> {
        &mut self.buf
    }

    fn from_buf(row: Self::Dim, buf: Vec<N>) -> Self {
        Self {
            buf,
            row
        }
    }
}

impl<N: Num> LinAlg<N> for Vector<N> where Vector<N>: LinAlgMul<N> 
{}

impl<N: Num> Vector<N> {
    pub fn to_matrix(&self) -> Matrix<N> {
        Matrix::from_buf(self.shape(), self.buf.clone())
    }

    pub fn diagonal(&self) -> Matrix<N> {
        Matrix::from_map((self.row, self.row), |(r, c)| {
            if r == c {
                self[r]
            }
            else {
                N::zero()
            }
        })
    }

    pub fn from_arr<const R: usize>(arr: [N; R]) -> Self {
        Self {
            buf: arr.to_vec(),
            row: R,
        }
    }

    pub fn one_hot(row: usize, hot: usize) -> Self {
        let mut buf = vec![N::zero(); row];
        buf[hot] = N::one();

        Self {
            buf,
            row
        }
    }
}