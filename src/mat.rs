use std::ops::{Index, IndexMut};
use std::marker::PhantomData;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;
use super::num::*;

/// type representing matrix dimensions
pub type Dim = (usize, usize);

pub trait MatrixType<T: Num> 
where
    Self: Index<Dim, Output=T> + Index<usize, Output=T> + Sized
{
    /// Returns the number of rows
    fn row(&self) -> usize;
    /// Returns the number of columns
    fn col(&self) -> usize;
    /// Returns the row stride 
    fn stride(&self) -> usize; 
    
    /// Returns the dimensions (row, col)
    fn dim(&self) -> Dim {
        (self.row(), self.col())
    }
    
    /// Returns the inverse dimensions (col, row)
    fn dim_inv(&self) -> Dim {
        (self.col(), self.row())
    }  
    
    /// Returns whether col == row
    fn is_square(&self) -> bool {
        self.row() == self.col()
    }

    /// Returns whether row == 1
    fn is_row(&self) -> bool {
        self.row() == 1
    }

    /// Returns whether column == 1
    fn is_col(&self) -> bool {
        self.col() == 1
    }
    
    /// Converts a (x, y) coordinate to buffer index
    fn to_index(&self, (r, c): Dim) -> usize {
        self.stride() * r + c
    }

    /// Returns a shared iterator to the matrix
    fn iter(&self) -> MatrixIter<'_, T, Self> {
        MatrixIter::new(self)
    }

    /// Maps each element by given function
    fn map<F>(&self, f: F) -> Matrix<T>
    where
        F: Fn(T) -> T
    {
        let buf = self
            .iter()
            .map(|n| f(n))
            .collect();

        Matrix::from_buf(self.dim(), buf)
    }

    /// Scales each element by given scalar
    fn scale<F>(&self, scalar: T) -> Matrix<T>
    where
        F: Fn(T) -> T
    {
        let buf = self
            .iter()
            .map(|i| scalar * i)
            .collect();

        Matrix::from_buf(self.dim(), buf)
    }
    

    /// Returns a (n, 1) dimensioned matrix as a (n, n)
    /// matrix who's elements are mapped across its diagonal
    fn diagonal(&self) -> Matrix<T> {
        if self.col() > 1 {
            panic!()
        }

        Matrix::from_map((self.row(), self.row()), |(r, c)| {
            if c == r { 
                self[(r, 0)] 
            } 
            else {
                T::zero()
            }
        })
    }
    
    /// Returns the determinant of a square matrix
    fn determinant(&self) -> T {
        if !self.is_square() {
            panic!("determinant of non-square matrix")
        }

        if self.col() == 2 {
            return self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)];
        }

        (0..self.row()).fold(T::zero(), |acc, i| {
            self.minor((0, i))
                * self[(0, i)]
                * if i % 2 == 0 { T::one() } else { -T::one() }
                + acc
        })
    }
    
    /// Returns the minor of a given matrix element
    fn minor(&self, (row, col): (usize, usize)) -> T {
        if !self.is_square() {
            panic!("minor of non-square matrix")
        }

        Matrix::from_map((self.row()-1, self.col()-1), |(r, c)| {
            let ro = if r < row { 0 } else { 1 };
            let co = if c < col { 0 } else { 1 };
            self[(r + ro, c + co)]
        })
        .determinant()
    }
    
    /// Returns the cofactor a square matrix
    fn cofactor(&self) -> Matrix<T> {
        if !self.is_square() {
            panic!("cofactor of non-square matrix")
        }

        Matrix::from_map(self.dim(), |(r, c)| {
            self.minor((r, c))
                * if (r + c) % 2 == 0 { T::one()} else { -T::one() }
        })
    }
    
    /// Returns the inverse of a square matrix
    fn inverse(&self) -> Matrix<T> {
        let det = self.determinant();

        if det == T::zero() {
            panic!("inverse of 0 determinant matrix")
        }
        
        self
            .cofactor()
            .transpose()
            .scale(T::one() / det)
    }

    /// Returns the sum of two matrices
    fn add(&self, other: &Self) -> Matrix<T> {
        if self.dim() != other.dim() {
            panic!("addition error, {:?} by {:?}", self.dim(), other.dim())
        }

        Matrix::from_map(self.dim(), |pos| self[pos] + other[pos])
    }

    /// Returns the difference of two matrices
    fn sub(&self, other: &Self) -> Matrix<T> {
        if self.dim() != other.dim() {
            panic!("subtraction error, {:?} by {:?}", self.dim(), other.dim())
        }

        Matrix::from_map(self.dim(), |pos| self[pos] - other[pos])    
    }

    /// Returns the product of two matrices
    fn mul(&self, other: &Self) -> Matrix<T> {
        if self.col() != other.row() {
            panic!("multipication error, {:?} by {:?}", self.dim(), other.dim())
        }

        let mut buf = Vec::with_capacity(self.row()*other.col());

        for row in 0..self.row() {
            for col in 0..other.col() {
                let acc = (0..self.stride()).fold(T::zero(), |acc, i| {
                    acc + self[(row, i)] * other[(i, col)]
                });
                buf.push(acc);
            }
        }

        Matrix::from_buf((self.row(), other.col()), buf)
    }
}

#[derive(Clone)]
pub struct Matrix<T: Num=N> {
    buf: Vec<T>,
    row: usize,
    col: usize,
}

impl<T: Num> Matrix<T> {
    /// Returns a new (r, c) matrix from 2D [[c]; r] array
    pub fn from_arr<const C: usize, const R: usize>(arr: [[T; C]; R]) -> Self {
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

    /// Returns new matrix with set buffer
    pub fn from_buf((row, col): Dim, buf: Vec<T>) -> Self {
        if buf.len() != row*col {
            panic!()
        }
        
        Self {
            buf,
            row,
            col
        }
    }

    /// Returns a new matrix mapped by (r, c) index
    pub fn from_map<F>((row, col): Dim, mut map: F) -> Self 
    where
        F: FnMut(Dim) -> T
    {
        let buf = (0..row*col)
            .map(|i| map((i/col, i%col)))
            .collect();

        Self {
            buf,
            row,
            col
        }
    }

    /// Returns a new identity matrix 
    pub fn identity(len: usize) -> Self {
        Self::from_map((len, len), |(r, c)| {
            if c == r { 
                T::one() 
            } 
            else { 
                T::zero() 
            }
        })
    }
    
    /// Returns a new matrix with all elements set to fill
    pub fn fill((row, col): (usize, usize), fill: T) -> Self {
        Self::from_map((row, col), |_| fill)
    }

    /// Returns a new matrix with all elements set to 0
    pub fn zeros((row, col): (usize, usize)) -> Self {
        Self::from_map((row, col), |_| T::zero())
    }

    /// Returns a new (0, 0) matrix
    pub fn empty() -> Self {
        Self {
            buf: Vec::new(),
            col: 0,
            row: 0,
        }
    }

    /// Returns a new matrix with random values between [-1, 1]
    pub fn random((row, col): (usize, usize)) -> Self 
    where 
        Standard: Distribution<T>
    {
        let mut rng = rand::thread_rng();
        Self::from_map((row, col), |_| rng.gen_range(-T::one()..T::one()))
    }

    /// Returns a new transpose matrix
    pub fn transpose(&self) -> Self {
        self.slice(self.dim_inv(), |(r, c)| (c, r)).to_matrix()
    }

    /// Returns a new scaled matrix
    pub fn scale(&self, scalar: T) -> Self {
        Self::from_map(self.dim(), |(r, c)| scalar * self[(r, c)])
    }

    /// Returns a new mapped matrix slice
    pub fn slice<F>(&self, dim: Dim, map: F) -> MatrixSlice<'_, T, impl Fn(Dim) -> Dim> 
    where
        F: Fn(Dim) -> Dim
    {
        MatrixSlice::new(self, dim, map)
    }

    /// Returns a slice to the matrix's rows
    pub fn rows(&self, beg: usize, num: usize, stride: usize) -> MatrixSlice<'_, T, impl Fn(Dim) -> Dim> {
        self.slice((num, self.col), move |(i, j)| (beg + i*stride, j))
    }

    /// Returns a slice to the matrix's columns
    pub fn cols(&self, beg: usize, num: usize, stride: usize) -> MatrixSlice<'_, T, impl Fn(Dim) -> Dim> {
        self.slice((num, self.row), move |(i, j)| (j, beg + i*stride))
    }

    /// Returns a slice of the transpose matrix
    pub fn transposed(&self) -> MatrixSlice<'_, T, impl Fn(Dim) -> Dim> {
        self.slice(self.dim_inv(), |(r, c)| (c, r))
    }

    pub fn filled(&mut self, fill: T) -> &mut Self {
        for n in self.buf.iter_mut() {
            *n = fill;
        }
        self
    }

    pub fn add_eq<M: MatrixType<T>>(&mut self, other: &M) -> &Self {
        if self.dim() != other.dim() {
            panic!("addition error")
        }

        for (i, n) in self.buf.iter_mut().enumerate() {
            *n += other[i]
        }

        self
    }

    pub fn sub_eq<M: MatrixType<T>>(&mut self, other: &M) -> &Self{
        if self.dim() != other.dim() {
            panic!("addition error")
        }

        for (i, n) in self.buf.iter_mut().enumerate() {
            *n -= other[i]
        }

        self
    }
}

impl<T: Num> MatrixType<T> for Matrix<T> {
    fn row(&self) -> usize {
        self.row
    }

    fn col(&self) -> usize {
        self.col
    }

    fn stride(&self) -> usize {
        self.col
    }
}

impl<'a, T: Num> Index<Dim> for Matrix<T> {
    type Output = T;

    fn index(&self, i: Dim) -> &Self::Output {
        &self.buf[self.to_index(i)]
    }
}

impl<'a, T: Num> IndexMut<Dim> for Matrix<T> {
    fn index_mut(&mut self, i: Dim) -> &mut Self::Output {
        let flattened = self.to_index(i);
        &mut self.buf[flattened]
    }
}

impl<'a, T: Num> Index<usize> for Matrix<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        &self.buf[i]
    }
}

impl<'a, T: Num> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.buf[i]
    }
}

pub struct MatrixSlice<'a, T, F> 
where
    T: Num,
    F: Fn(Dim) -> Dim 
{
    mat: &'a Matrix<T>,
    row: usize,
    col: usize,
    map: F,
}

impl<'a, T, F> MatrixType<T> for MatrixSlice<'a, T, F> 
where
    T: Num,
    F: Fn(Dim) -> Dim 
{
    fn row(&self) -> usize {
        self.row
    }

    fn col(&self) -> usize {
        self.col
    }
    
    fn stride(&self) -> usize {
        self.mat.col
    }
}

impl<'a, T, F> MatrixSlice<'a, T, F> 
where
    T: Num,
    F: Fn(Dim) -> Dim 
{
    fn to_index(&self, (r, c): Dim) -> usize {
        self.mat.col() * r + c
    }

    pub fn new(mat: &'a Matrix<T>, (row, col): Dim, map: F) -> MatrixSlice<'a, T, impl Fn(Dim) -> Dim> {
        MatrixSlice { 
            mat, 
            row, 
            col, 
            map
        }
    }

    pub fn to_matrix(&self) -> Matrix<T> {
        let buf = (0..self.row*self.col)
            .map(|i| self[(i/self.col, i%self.col)])
            .collect();

        Matrix::from_buf(self.dim(), buf)
    }
}

impl<'a, T, F> Index<Dim> for MatrixSlice<'a, T, F> 
where
    T: Num,
    F: Fn(Dim) -> Dim 
{
    type Output=T;

    fn index(&self, i: Dim) -> &Self::Output {
        &self.mat[self.to_index((self.map)(i))]
    }
}

impl<'a, T, F> Index<usize> for MatrixSlice<'a, T, F> 
where
    T: Num,
    F: Fn(Dim) -> Dim 
{
    type Output=T;

    fn index(&self, i: usize) -> &Self::Output {
        let i = (self.map)((i/self.col, i%self.col));
        &self.mat[self.to_index(i)]
    }
}

pub struct MatrixIter<'a, T, M> 
where
    T: Num,
    M: MatrixType<T>
{
    mat: &'a M,
    i: usize,
    _t: PhantomData<T>
}

impl<'a, T, M> MatrixIter<'a, T, M> 
where
    T: Num,
    M: MatrixType<T>
{
    pub fn new(mat: &'a M) -> Self {
        Self { 
            mat, 
            i: 0, 
            _t: PhantomData::default() 
        }
    }
}

impl<'a, T, M> Iterator for MatrixIter<'a, T, M> 
where
    T: Num,
    M: MatrixType<T>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.mat.row() * self.mat.col() {
            return None
        }

        self.i += 1;
        Some(self.mat[self.i - 1])
    }
}