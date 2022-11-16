use crate::mat::{MatrixType, Dim};
use crate::num::Num;
use std::marker::PhantomData;
use std::ops::Index;


pub struct OneHot<T: Num> {
    row: usize,
    hot: usize,
    one: T,
    zero: T,
    _t: PhantomData<T>
}

impl<T: Num> Index<Dim> for OneHot<T> {
    type Output = T;

    fn index(&self, (row, _): Dim) -> &Self::Output {
        if row == self.hot {
            &self.one
        } 
        else {
            &self.zero
        }
    }
}

impl<T: Num> Index<usize> for OneHot<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        if i == self.hot {
            &self.one
        } 
        else {
            &self.zero
        }
    }
}

impl<T: Num> MatrixType<T> for OneHot<T> {
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