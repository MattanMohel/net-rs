use std::ops::Add;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;
use std::fmt;

use super::num::N;
use super::num::Num;

#[derive(Debug)]
pub enum MatErr {
    Mul,
    Add,
    Sqr,
    Inv
}

#[derive(Clone)]
pub struct Mat<T: Num = N> {
    buf: Vec<T>,
    row: usize,
    col: usize,
}

impl<T: Num> Mat<T> {
    pub fn empty() -> Self {
        Self {
            buf: Vec::new(),
            col: 0,
            row: 0,
        }
    }

    pub fn from_arr<const C: usize, const R: usize>(arr: [[T; C]; R]) -> Self {
        let buf = arr.iter().flatten().cloned().collect();

        Self {
            buf,
            col: C,
            row: R,
        }
    }

    pub fn from_vec((row, col): (usize, usize), buf: Vec<T>) -> Self {
        Self { buf, row, col }
    }

    pub fn from_fn<F>((row, col): (usize, usize), f: F) -> Self
    where
        F: Fn() -> T,
    {
        let buf = (0..row*col).map(|_| f()).collect();

        Self { buf, row, col }
    }

    pub fn from_map<F>((row, col): (usize, usize), map: F) -> Self
    where
        F: Fn(usize, usize) -> T,
    {
        let buf = (0..row*col).map(|i| map(i / col, i % col)).collect();

        Self { buf, row, col }
    }

    pub fn fill((row, col): (usize, usize), fill: T) -> Self {
        Self::from_fn((row, col), || fill)
    }

    pub fn zeros((row, col): (usize, usize)) -> Self {
        Self::from_fn((row, col), || T::zero())
    }

    pub fn identity(len: usize) -> Self {
        Self::from_map((len, len), |r, c| if c == r { T::one() } else { T::zero() })
    }

    pub fn transpose(&self) -> Self {
        Self::from_map((self.row, self.col), |r, c| self[(c, r)])
    }

    pub fn cofactor(&self) -> Result<Self, MatErr> {
        if !self.is_square() {
            return Err(MatErr::Sqr);
        }

        Ok(Self::from_map(self.dim(), |r, c| {
            self.minor((r, c)).unwrap().determinant().unwrap()
                * if (r + c) % 2 == 0 { T::one()} else { T::neg() }
        }))
    }

    pub fn adjoint(&self) -> Result<Self, MatErr> {
        Ok(self.cofactor()?.transpose())
    }

    pub fn inverse(&self) -> Result<Self, MatErr> {
        let det = self.determinant()?;

        if det == T::zero() {
            return Err(MatErr::Inv)
        }
        
        Ok(self.adjoint()?.scale(det.inv()))
    }

    pub fn determinant(&self) -> Result<T, MatErr> {
        if !self.is_square() {
            return Err(MatErr::Sqr);
        }

        if self.col == 2 {
            return Ok(self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]);
        }

        Ok((0..self.row).fold(T::zero(), |acc, i| {
            self.minor((0, i)).unwrap().determinant().unwrap()
                * self[(0, i)]
                * if i % 2 == 0 { T::one() } else { T::neg() }
                + acc
        }))
    }

    pub fn minor(&self, (row, col): (usize, usize)) -> Result<Self, MatErr> {
        if !self.is_square() {
            return Err(MatErr::Sqr);
        }

        Ok(Self::from_map((self.row - 1, self.col - 1), |r, c| {
            let ro = if r < row { 0 } else { 1 };
            let co = if c < col { 0 } else { 1 };
            self[(r + ro, c + co)]
        }))
    }

    pub fn to_index(&self, (row, col): (usize, usize)) -> usize {
        self.col * row + col
    }

    pub fn to_coord(&self, index: usize) -> (usize, usize) {
        (index / self.col, index % self.col)
    }

    pub fn dim(&self) -> (usize, usize) {
        (self.row, self.col)
    }

    pub fn row(&self) -> usize {
        self.row
    }
    
    pub fn col(&self) -> usize {
        self.col
    }

    pub fn is_square(&self) -> bool {
        self.col == self.row
    }

    pub fn buf(&self) -> &Vec<T> {
        &self.buf
    }

    pub fn buf_mut(&mut self) -> &mut Vec<T> {
        &mut self.buf
    }

    pub fn rows(&self) -> Rows<'_, T> {
        Rows::from(self)
    }
        
    pub fn get_row(&self, row: usize) -> Vec<T> {  
        (0..self.col)
            .map(|i| self.buf[row * self.col + i])
            .collect()
    }

    pub fn cols(&self) -> Cols<'_, T> {
        Cols::from(self)
    }

    pub fn get_col(&self, col: usize) -> Vec<T> {
        (0..self.row)
            .map(|i| self.buf[col + i * self.col])
            .collect()
    }    

    pub fn iter(&self) -> MatIter<'_, T> {
        MatIter::from(self)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, MatErr> {
        if self.dim() != rhs.dim() {
            return Err(MatErr::Add);
        }

        Ok(Self::from_map(self.dim(), |r, c| {
            self[(r, c)] + rhs[(r, c)]
        }))
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, MatErr> {
        if self.col != rhs.row {
            return Err(MatErr::Mul);
        }

        Ok(Self::from_map((self.row, rhs.col), |r, c| {
            self
                .get_row(r)
                .iter()
                .zip(rhs.get_col(c).iter())
                .fold(T::zero(), |acc, (v1, v2)| acc + (*v1) * (*v2))
        }))
    }

    pub fn scale(&self, scalar: T) -> Self {
        Self::from_map(self.dim(), |c, r| scalar * self[(c, r)])
    }
}

impl<T: Num> Index<(usize, usize)> for Mat<T> {
    type Output = T;

    fn index(&self, (col, row): (usize, usize)) -> &Self::Output {
        &self.buf[self.row * col + row]
    }
}

impl<T: Num> fmt::Display for Mat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let buf = self.rows().fold(String::new(), |acc, row| format!("{}\n{:?}", acc, row));
        write!(f, "{}", buf)
    }
}


/// Iterates Matrix rows by element
pub struct MatIter<'m, T: Num> {
    mat: &'m Mat<T>,
    i: usize,
}

impl<'m, T: Num> From<&'m Mat<T>> for MatIter<'m, T> {
    fn from(mat: &'m Mat<T>) -> Self {
        MatIter { mat, i: 0 }
    }
}

impl<'m, T: Num> Iterator for MatIter<'m, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        Some(self.mat.buf()[self.i - 1])
    }
}

/// Iterates Matrix by columns
pub struct Cols<'m, T: Num> {
    mat: &'m Mat<T>,
    i: usize,
}

impl<'m, T: Num> From<&'m Mat<T>> for Cols<'m, T> {
    fn from(mat: &'m Mat<T>) -> Self {
        Cols { mat, i: 0 }
    }
}

impl<'m, T: Num> Iterator for Cols<'m, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.mat.col() {
            return None;
        }
        
        self.i += 1;
        Some(self.mat.get_col(self.i - 1))
    }
}

pub struct Rows<'m, T: Num> {
    mat: &'m Mat<T>,
    i: usize,
}

impl<'m, T: Num> From<&'m Mat<T>> for Rows<'m, T> {
    fn from(mat: &'m Mat<T>) -> Self {
        Rows { mat, i: 0 }
    }
}

impl<'m, T: Num> Iterator for Rows<'m, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.mat.row() {
            return None;
        }
        
        self.i += 1;
        Some(self.mat.get_row(self.i - 1))
    }
}