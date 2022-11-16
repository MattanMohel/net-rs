use std::ops::Index;
use super::num::Num;
use super::matrix::Dim;
use super::matrix::Matrix;
use super::matrix::IMatrix;

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

impl<'a, T, F> IMatrix<T> for MatrixSlice<'a, T, F> 
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
}

impl<'a, T, F> MatrixSlice<'a, T, F> 
where
    T: Num,
    F: Fn(Dim) -> Dim 
{
    /// Returns new slice from map
    pub fn new(mat: &'a Matrix<T>, (row, col): Dim, map: F) -> MatrixSlice<'a, T, impl Fn(Dim) -> Dim> {
        MatrixSlice { 
            mat, 
            row, 
            col, 
            map
        }
    }

    /// Collects slice into an owned matrix
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
