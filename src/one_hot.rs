use crate::matrix::Matrix;

use super::matrix::IMatrix;
use super::matrix::Dim;
use super::num::N;
use super::num::Num;
use std::marker::PhantomData;
use std::ops::Index;


pub struct OneHot<T: Num=N> {
    row: usize,
    hot: usize,
    values: [T; 2],
    _t: PhantomData<T>
}

impl<T: Num> OneHot<T> {
    pub fn new(row: usize, hot: usize) -> Self {
        Self {
            row,
            hot,
            values: [T::zero(), T::one()],
            _t: PhantomData::default() 
        }
    }

    pub fn hot(&self) -> usize {
        self.hot
    }
}

impl<T: Num> Index<Dim> for OneHot<T> {
    type Output = T;

    fn index(&self, (row, _): Dim) -> &Self::Output {
        if row == self.hot {
            &self.values[1]
        } 
        else {
            &self.values[0]
        }
    }
}

impl<T: Num> Index<usize> for OneHot<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        if i == self.hot {
            &self.values[1]
        } 
        else {
            &self.values[0]
        }
    }
}

impl<T: Num> IMatrix<T> for OneHot<T> {
    fn row(&self) -> usize {
        self.row
    }

    fn col(&self) -> usize {
        1
    }

    fn stride(&self) -> usize {
        1
    }

    fn to_matrix(&self) -> Matrix<T> {
        let buf = (0..self.row)
            .map(|i| {
                if i == self.hot {
                    T::one()
                }
                else {
                    T::zero()
                }
            })
            .collect();

        Matrix::from_buf(self.dim(), buf)
    }
} 