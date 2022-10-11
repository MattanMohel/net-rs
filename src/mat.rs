use std::iter::Map;
use std::iter::Skip;
use std::iter::StepBy;
use std::iter::Take;
use std::ops::Add;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Range;
use std::ops::Sub;
use std::fmt;
use std::slice::Iter;

use super::num::N;
use super::num::Num;

use rand::distributions::Standard;
use rand::prelude::*;

#[derive(Debug)]
pub enum MatErr {
    Mul,
    Add,
    Sqr,
    Inv,
    Dim
}

pub fn vec_map<F, T>(len: usize, f: F)  -> Vec<T>
where 
    F: Fn(usize) -> T
{
    (0..len)
        .map(|i| f(i))
        .collect()
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
        let buf = arr.iter().copied().flatten().collect();

        Self {
            buf,
            col: C,
            row: R
        }
    }

    pub fn as_row(&self, row: usize) -> Result<Self, MatErr> {
        if self.row != 1 {
            return Err(MatErr::Dim)
        }

        let buf = self.buf.iter().copied().cycle().take(row * self.col).collect();
    
        Ok(Self { 
            buf,
            row, 
            col: self.col
        })
    }

    pub fn as_col<const R: usize>(&self, col: usize) -> Result<Self, MatErr> {
        if self.col != 1 {
            return Err(MatErr::Dim)
        }

        let buf = self.buf.iter().copied().flat_map(|n| std::iter::repeat(n).take(col)).collect();
        
        Ok(Self { 
            buf,
            col,
            row: self.row
        })
    }

    pub fn from_vec((row, col): (usize, usize), buf: Vec<T>) -> Self {
        Self { buf, row, col }
    }

    pub fn from_fn<F>((row, col): (usize, usize), mut f: F) -> Self
    where
        F: FnMut() -> T,
    {
        let buf = (0..row*col).map(|_| f()).collect();

        Self { buf, row, col }
    }

    pub fn from_map<F>((row, col): (usize, usize), mut map: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
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

    pub fn random((row, col): (usize, usize), min: T, max: T) -> Self 
    where 
        Standard: Distribution<T>
    {
        rand::random();
        let mut rng = rand::thread_rng();
        Self::from_fn((row, col), || rng.gen() * (max - min) + min)
    }

    pub fn identity(len: usize) -> Self {
        Self::from_map((len, len), |r, c| if c == r { T::one() } else { T::zero() })
    }

    pub fn as_identity(&self) -> Result<Self, MatErr> {
        if self.row == 1 {
            Ok(Self::from_map((self.col, self.col), |r, c| {
                if c == r { 
                    self[(1, c)] 
                } 
                else {
                    T::zero()
                }
            }))
        }
        else if self.col == 1 {
            Ok(Self::from_map((self.row, self.row),  |r, c| {
                if c == r { 
                    self[(c, 1)] 
                } 
                else { 
                    T::zero() 
                }
            }))
        }
        else {
            Err(MatErr::Dim)
        } 
    }

    pub fn transpose(&self) -> Self {
        Self::from_map((self.col, self.row), |r, c| self[(c, r)])
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
        
        Ok(self.adjoint()?.scaled(det.inv()))
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

    pub fn rows(&self) -> usize {
        self.row
    }
    
    pub fn cols(&self) -> usize {
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
        
    pub fn row(&self, row: usize) -> Take<Skip<Iter<T>>> {
        self.buf.iter().skip(row * self.col).take(self.col)
    }

    pub fn col(&self, col: usize) -> Take<StepBy<Skip<Iter<T>>>> {
        self.buf.iter().skip(col).step_by(self.col).take(self.row)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, MatErr> {
        if self.dim() != rhs.dim() {
            return Err(MatErr::Add);
        }

        Ok(Self::from_map(self.dim(), |r, c| {
            self[(r, c)] + rhs[(r, c)]
        }))
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, MatErr> {
        if self.dim() != rhs.dim() {
            return Err(MatErr::Add);
        }

        Ok(Self::from_map(self.dim(), |r, c| {
            self[(r, c)] - rhs[(r, c)]
        }))
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, MatErr> {
        if self.col != rhs.row {
            return Err(MatErr::Mul);
        }

        Ok(Self::from_map((self.row, rhs.col), |r, c| {
            self
                .row(r)
                .zip(rhs.col(c))
                .fold(T::zero(), |acc, (v1, v2)| acc + (*v1) * (*v2))
        }))
    }

    pub fn scaled(&self, scalar: T) -> Self {
        Self::from_map(self.dim(), |r, c| scalar * self[(r, c)])
    }

    pub fn mapped<F>(&self, mut map: F) -> Self 
    where
        F: FnMut(T) -> T
    {
        Self::from_map(self.dim(), |r, c| map(self[(r, c)]))
    }
}

impl<T: Num> Index<(usize, usize)> for Mat<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.buf[self.col * row + col]
    }
}

impl<T: Num> IndexMut<(usize, usize)> for Mat<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.buf[self.col * row + col]
    }
}

impl<T: Num> fmt::Display for Mat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let buf = {
            (0..self.row)
                .map(|i| self.row(i))
                .fold(String::new(), |acc, row| format!("{}\n{:?}", acc, row.copied().collect::<Vec<T>>()))
        };
        write!(f, "{}", buf)
    }
}

/// Iterates Matrix rows by element
pub struct Rows<'m, T: Num> {
    mat: &'m Mat<T>,
    i: usize,
}

impl<'m, T: Num> From<&'m Mat<T>> for Rows<'m, T> {
    fn from(mat: &'m Mat<T>) -> Self {
        Self { 
            mat,  
            i: 0
        }
    }
}

impl<'m, T: Num> Iterator for Rows<'m, T> {
    type Item = Take<Skip<Iter<'m, T>>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.mat.row {
            return None
        }

        self.i += 1;    
        Some(self.mat.row(self.i - 1))
    }
}

pub struct Cols<'m, T: Num> {
    mat: &'m Mat<T>,
    i: usize,
}

impl<'m, T: Num> From<&'m Mat<T>> for Cols<'m, T> {
    fn from(mat: &'m Mat<T>) -> Self {
        Self { 
            mat,  
            i: 0
        }
    }
}

impl<'m, T: Num> Iterator for Cols<'m, T> {
    type Item = Take<StepBy<Skip<Iter<'m, T>>>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.mat.col {
            return None
        }

        self.i += 1;    
        Some(self.mat.col(self.i - 1))
    }
}
