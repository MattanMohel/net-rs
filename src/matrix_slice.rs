use std::ops::Index;
use super::num::Num;
use super::num::N;
use super::matrix::Dim;
use super::matrix::Matrix;
use super::matrix::IMatrix;

pub struct MatrixSlice<'a, F, T=N> 
where
    T: Num,
    F: Fn(Dim) -> Dim 
{
    mat: &'a Matrix<T>,
    row: usize,
    col: usize,
    map: F,
}

impl<'a, F, T> IMatrix<T> for MatrixSlice<'a, F, T> 
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
        self.mat.col()
    }

    fn to_matrix(&self) -> Matrix<T> {
        let buf = (0..self.row*self.col)
            .map(|i| self[(i/self.col, i%self.col)])
            .collect();

        Matrix::from_buf(self.dim(), buf)    }
}

impl<'a, F, T> MatrixSlice<'a, F, T> 
where
    T: Num,
    F: Fn(Dim) -> Dim 
{
    pub fn new(mat: &'a Matrix<T>, (row, col): Dim, map: F) -> MatrixSlice<'a, impl Fn(Dim) -> Dim, T> {
        MatrixSlice { 
            mat, 
            row, 
            col, 
            map
        }
    }
}

impl<'a, F, T> Index<Dim> for MatrixSlice<'a, F, T>
where
    T: Num,
    F: Fn(Dim) -> Dim 
{
    type Output=T;

    fn index(&self, i: Dim) -> &Self::Output {
        &self.mat[self.to_index((self.map)(i))]
    }
}

impl<'a, F, T> Index<usize> for MatrixSlice<'a, F, T>
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