use std::{ops::Index, marker::PhantomData};
use super::num::*;

/// type representing matrix dimensions
type Dim = (usize, usize);

pub trait MatrixType<T: Num> 
where
    Self: Index<Dim, Output=T> + Index<usize, Output=T> + Sized
{
    fn row(&self) -> usize;
    fn col(&self) -> usize;
    fn buf(&self) -> &Vec<T>;
    fn buf_stride(&self) -> usize;

    fn to_index(&self, (r, c): Dim) -> usize {
        self.buf_stride() * r + c
    }

    fn dim(&self) -> Dim {
        (self.row(), self.col())
    }

    fn dim_inv(&self) -> Dim {
        (self.col(), self.row())
    }

    fn is_square(&self) -> bool {
        self.row() == self.col()
    }

    fn iter(&self) -> MatrixIterator<'_, T, Self> {
        MatrixIterator::new(self)
    }
    
    fn add(&self, other: &Self) -> Matrix<T> {
        if self.dim() != other.dim() {
            panic!()
        }

        Matrix::from_map(self.dim(), |pos| self[pos] + other[pos])
    }

    fn sub(&self, other: &Self) -> Matrix<T> {
        if self.dim() != other.dim() {
            panic!()
        }

        Matrix::from_map(self.dim(), |pos| self[pos] - other[pos])    }
}

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

    pub fn from_map<F>((row, col): Dim, map: F) -> Self 
    where
        F: Fn(Dim) -> T
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

    fn buf_stride(&self) -> usize {
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

    fn buf_stride(&self) -> usize {
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
        &self.mat[self.to_index((self.map)((i/self.col, i%self.col)))]
    }
}

pub struct MatrixIterator<'a, T, M> 
where
    T: Num,
    M: MatrixType<T>
{
    mat: &'a M,
    i: usize,
    _t: PhantomData<T>
}

impl<'a, T, M> MatrixIterator<'a, T, M> 
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

impl<'a, T, M> Iterator for MatrixIterator<'a, T, M> 
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