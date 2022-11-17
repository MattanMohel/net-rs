use super::matrix::{IMatrix, Dim};
use super::num::Num;
use std::marker::PhantomData;
use std::ops::Index;


pub struct OneHot<T: Num> {
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
} 