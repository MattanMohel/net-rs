use std::{ops::Index, marker::PhantomData};
use rand::{distributions::Standard, prelude::Distribution, Rng};

use super::num::*;

/// type representing matrix dimensions
type Dim = (usize, usize);

pub trait MatrixType<T: Num> 
where
    Self: Index<Dim, Output=T> + Index<usize, Output=T> + Sized
{
    /// Returns the number of rows
    fn row(&self) -> usize;
    /// Returns the number of columns
    fn col(&self) -> usize;
    /// Returns a reference to the buffer
    fn buf(&self) -> &Vec<T>;
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
    
    /// Returns wether col == row
    fn is_square(&self) -> bool {
        self.row() == self.col()
    }
    
    /// Converts an (x, y) coordinate into a buffer index
    fn to_index(&self, (r, c): Dim) -> usize {
        self.stride() * r + c
    }

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

    fn iter(&self) -> MatrixIter<'_, T, Self> {
        MatrixIter::new(self)
    }

    fn map<F>(&self, f: F) -> Matrix<T>
    where
        F: Fn(T) -> T
    {
        let buf = 
            self
                .iter()
                .map(|i| f(i))
                .collect();

        Matrix::from_buf(self.dim(), buf)
    }

    fn scale<F>(&self, scalar: T) -> Matrix<T>
    where
        F: Fn(T) -> T
    {
        let buf = 
            self
                .iter()
                .map(|i| scalar * i)
                .collect();

        Matrix::from_buf(self.dim(), buf)
    }
    
    fn add(&self, other: &Self) -> Matrix<T> {
        if self.dim() != other.dim() {
            panic!("addition error")
        }

        Matrix::from_map(self.dim(), |pos| self[pos] + other[pos])
    }

    fn sub(&self, other: &Self) -> Matrix<T> {
        if self.dim() != other.dim() {
            panic!("subtraction error")
        }

        Matrix::from_map(self.dim(), |pos| self[pos] - other[pos])    
    }

    fn mul(&self, other: &Self) -> Matrix<T> {
        if self.col() != other.row() {
            panic!("multipication error")
        }

        let mut buf = Vec::with_capacity(self.row()*other.col());

        for row in 0..self.row() {
            for col in 0..other.col() {
                let acc = (0..self.stride()).fold(T::zero(), |acc, i| acc + self[(row, i)] * other[(i, col)]);
                buf.push(acc);
            }
        }

        Matrix::from_buf((self.row(), other.col()), buf)
    }
}

#[derive(Clone)]
pub struct Matrix<T: Num> {
    buf: Vec<T>,
    row: usize,
    col: usize,
}

impl<T: Num> Matrix<T> {
    pub fn from_arr<const C: usize, const R: usize>(arr: [[T; C]; R]) -> Self {
        let buf = 
            arr
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

    pub fn from_map<F>((row, col): Dim, mut map: F) -> Self 
    where
        F: FnMut(Dim) -> T
    {
        let buf = 
            (0..row*col)
                .map(|i| map((i/col, i%col)))
                .collect();

        Self {
            buf,
            row,
            col
        }
    }

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
    
    pub fn fill((row, col): (usize, usize), fill: T) -> Self {
        Self::from_map((row, col), |_| fill)
    }

    pub fn zeros((row, col): (usize, usize)) -> Self {
        Self::from_map((row, col), |_| T::zero())
    }

    pub fn empty() -> Self {
        Self {
            buf: Vec::new(),
            col: 0,
            row: 0,
        }
    }

    pub fn random((row, col): (usize, usize), min: T, max: T) -> Self 
    where 
        Standard: Distribution<T>
    {
        let mut rng = rand::thread_rng();
        Self::from_map((row, col), |_| rng.gen() * (max - min) + min)
    }

    pub fn slice<F>(&self, dim: Dim, map: F) -> MatrixSlice<'_, T, impl Fn(Dim) -> Dim> 
    where
        F: Fn(Dim) -> Dim
    {
        MatrixSlice::new(self, dim, map)
    }

    pub fn rows(&self, beg: usize, num: usize, stride: usize) -> MatrixSlice<'_, T, impl Fn(Dim) -> Dim> {
        self.slice((num, self.col), move |(i, j)| (beg + i*stride, j))
    }

    pub fn cols(&self, beg: usize, num: usize, stride: usize) -> MatrixSlice<'_, T, impl Fn(Dim) -> Dim> {
        self.slice((num, self.row), move |(i, j)| (j, beg + i*stride))
    }

    pub fn transposed(&self) -> MatrixSlice<'_, T, impl Fn(Dim) -> Dim> {
        self.slice(self.dim_inv(), |(r, c)| (c, r))
    }

    pub fn transpose(&self) -> Self {
        self.slice(self.dim_inv(), |(r, c)| (c, r)).to_matrix()
    }

    pub fn scale(&self, scalar: T) -> Self {
        Self::from_map(self.dim(), |(r, c)| scalar * self[(r, c)])
    }

    pub fn add_eq<M: MatrixType<T>>(&mut self, other: &M) {
        if self.dim() != other.dim() {
            panic!("addition error")
        }

        for (i, n) in self.buf.iter_mut().enumerate() {
            *n += other[i]
        }
    }

    pub fn sub_eq<M: MatrixType<T>>(&mut self, other: &M) {
        if self.dim() != other.dim() {
            panic!("addition error")
        }

        for (i, n) in self.buf.iter_mut().enumerate() {
            *n -= other[i]
        }
    }
}

impl<T: Num> MatrixType<T> for Matrix<T> {
    fn row(&self) -> usize {
        self.row
    }

    fn col(&self) -> usize {
        self.col
    }

    fn buf(&self) -> &Vec<T> {
        &self.buf
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

impl<'a, T: Num> Index<usize> for Matrix<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        &self.buf[i]
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

    fn buf(&self) -> &Vec<T> {
        &self.mat.buf
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
        let buf = 
            (0..self.row*self.col)
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