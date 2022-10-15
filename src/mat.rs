use std::iter::{
    Skip,
    Take,
    StepBy
};
use std::slice::{
    Iter,
    IterMut
};
use std::ops::{
    Index,
    IndexMut,
};
use super::num::{
    N,
    Num
};
use std::fmt;
use rand::prelude::*;
use rand::distributions::Standard;


/// Type representing a Matrix Error
pub type MatErr<T=()> = Result<T, MatErrType>;

#[derive(Debug)]
pub enum MatErrType {
    Mul,
    Add,
    Sqr,
    Inv,
    Dim
}

use MatErrType::*;

#[derive(Clone)]
pub struct Mat<T: Num = N> {
    buf: Vec<T>,
    row: usize,
    col: usize,
}

impl<T: Num> Mat<T> {
    /// Creates empty matrix with dimensions `(0, 0)`
    pub fn empty() -> Self {
        Self {
            buf: Vec::new(),
            col: 0,
            row: 0,
        }
    }

    /// Creates matrix from 2D-array with dimensions `(row, col)` where `row == R` and `col == C`
    pub fn from_arr<const C: usize, const R: usize>(arr: [[T; C]; R]) -> Self {
        let buf = arr.iter().copied().flatten().collect();

        Self {
            buf,
            col: C,
            row: R
        }
    }

    /// Creates matrix from buffer with dimensions `(row, col)`.
    /// ## Conditions
    /// - `row * col == buf.len()`
    pub fn from_vec((row, col): (usize, usize), buf: Vec<T>) -> MatErr<Mat<T>> {
        if row * col != buf.len() {
            return Err(Dim)
        }
        
        Ok(Self { 
            buf, 
            row, 
            col 
        })
    }

    // TODO: Add 'from_rows' & 'from_cols' which will map by accpeted vectors of columns
    // or rows rather than elements

    /// Creates matrix from map over dimensions `(row, col)` in row by column 
    /// order where each iteration intakes its index and outputs the cell value
    pub fn from_map<F>((row, col): (usize, usize), mut map: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let buf = (0..row*col).map(|i| map(i / col, i % col)).collect();

        Self { 
            buf, 
            row, 
            col 
        }
    }

    /// Creates matrix from map over dimensions `(row, col)` in row by collumn 
    /// order where each iteration takle no index and outputs the cell value
    pub fn from_fn<F>((row, col): (usize, usize), mut f: F) -> Self
    where
        F: FnMut() -> T,
    {
        let buf = (0..row*col).map(|_| f()).collect();

        Self { 
            buf, 
            row, 
            col 
        }
    }

    /// Creates matrix with dimensions `(row, col)` filled with zeroes
    pub fn zeros((row, col): (usize, usize)) -> Self {
        Self::from_fn((row, col), || T::zero())
    }
    
    /// Creates matrix with dimensions `(row, col)` filled with a value
    pub fn fill((row, col): (usize, usize), fill: T) -> Self {
        Self::from_fn((row, col), || fill)
    }

    /// Creates matrix with dimensions `(row, col)` filled with random values between `[min, max)`
    pub fn random((row, col): (usize, usize), min: T, max: T) -> Self 
    where 
        Standard: Distribution<T>
    {
        let mut rng = rand::thread_rng();
        Self::from_fn((row, col), || rng.gen() * (max - min) + min)
    }

    /// Creates identity matrix with dimensions `(len, len)`
    pub fn identity(len: usize) -> Self {
        Self::from_map((len, len), |r, c| {
            if c == r { 
                T::one() 
            } 
            else { 
                T::zero() 
            }
        })
    }

    /// Creates diagonal "identity" matrix filled by row values. Given
    /// the input `[1, 2, 3]` it outputs `[[1, 0, 0], [0, 2, 0], [0, 0, 3]]`.
    /// ## Conditions
    /// - `self.rows() == 1`    
    pub fn row_diagonal(&self) -> MatErr<Self> {
        if self.row != 1 {
            return Err(Dim)
        }

        Ok(Self::from_map((self.col, self.col), |r, c| {
            if c == r { 
                self[(1, c)] 
            } 
            else {
                T::zero()
            }
        }))
    }

    /// Creates diagonal "identity" matrix filled by column values. Given
    /// the input `[[1], [2], [3]]` it outputs `[[1, 0, 0], [0, 2, 0], [0, 0, 3]]`.
    /// ## Conditions
    /// - `self.col() == 1`   
    pub fn col_diagonal(&self) -> MatErr<Self> {
        if self.col != 1 {
            return Err(Dim)
        }

        Ok(Self::from_map((self.col, self.col), |r, c| {
            if c == r { 
                self[(r, 1)] 
            } 
            else {
                T::zero()
            }
        }))
    }

    /// Creates transpose matrix, rotated 90 degrees from the source matrix. Given
    /// the input `[1, 2, 3]` it outputs `[[1], [2], [3]]`, inversing all rows and columns
    pub fn transpose(&self) -> Self {
        Self::from_map((self.col, self.row), |r, c| self[(c, r)])
    }

    /// Creates cofactor matrix, setting each element to minor and negating it
    /// if its coordinate `(r, c)` doesn't satisfy `(r + c) % 2 != 0`.
    /// ## Conditions
    /// - `self.is_square()`
    pub fn cofactor(&self) -> MatErr<Self> {
        if !self.is_square() {
            return Err(Sqr);
        }

        Ok(Self::from_map(self.dim(), |r, c| {
            self.minor((r, c)).unwrap()
                * if (r + c) % 2 == 0 { T::one()} else { T::neg() }
        }))
    }

    /// Creates inverse matrix `A-1` from matrix `A` satisfying `A * A-1 = Identity`. 
    /// ## Conditions
    /// - `Self.determinant() != 0`
    pub fn inverse(&self) -> MatErr<Self> {
        let det = self.determinant()?;

        if det == T::zero() {
            return Err(Inv)
        }
        
        Ok(self.cofactor()?.transpose().scaled(det.inv()))
    }

    /// Returns determinant, the the `n`-dimensional surface representating the 
    /// transfomation of the matrix, where `n == row == col` 
    /// ## Conditions
    /// - `self.is_square()`
    pub fn determinant(&self) -> MatErr<T> {
        if !self.is_square() {
            return Err(Sqr);
        }

        if self.col == 2 {
            return Ok(self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]);
        }

        Ok((0..self.row).fold(T::zero(), |acc, i| {
            self.minor((0, i)).unwrap()
                * self[(0, i)]
                * if i % 2 == 0 { T::one() } else { T::neg() }
                + acc
        }))
    }

    /// Returns determinant of a minor matrix. Given the input `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`
    /// the minor at `(1, 2)` would return the determinant of the matrix `[[1, 2], [7, 8]]`. 
    /// ## Conditions
    /// - `self.is_square()`
    pub fn minor(&self, (row, col): (usize, usize)) -> MatErr<T> {
        if !self.is_square() {
            return Err(Sqr);
        }

        Self::from_map((self.row - 1, self.col - 1), |r, c| {
            let ro = if r < row { 0 } else { 1 };
            let co = if c < col { 0 } else { 1 };
            self[(r + ro, c + co)]
        })
        .determinant()
    }

    /// Returns the dimensions `(row, col)`
    pub fn dim(&self) -> (usize, usize) {
        (self.row, self.col)
    }

    /// Returns the row count
    pub fn rows(&self) -> usize {
        self.row
    }

    /// Returns the column count
    pub fn cols(&self) -> usize {
        self.col
    }

    /// Returns whether `row == col`
    pub fn is_square(&self) -> bool {
        self.col == self.row
    }

    /// Returns a shared reference to the matrix buffer
    pub fn buf(&self) -> &Vec<T> {
        &self.buf
    }

    /// Returns a mutable reference to the matrix buffer
    pub fn buf_mut(&mut self) -> &mut Vec<T> {
        &mut self.buf
    }
    
    /// Returns a shared buffer iterator, indexing in row by column order
    pub fn iter(&self) -> Iter<'_, T> {
        self.buf.iter()
    }

    /// Returns a mutable buffer iterator, indexing in row by column order
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.buf.iter_mut()
    } 

    /// Returns a shared rows iterator
    pub fn iter_row(&self) -> Rows<'_, T> {
        Rows::from(self)
    }

    /// Returns a shared columns iterator
    pub fn iter_col(&self) -> Cols<'_, T> {
        Cols::from(self)
    }
    
    /// Returns the `row`th row
    pub fn row(&self, row: usize) -> Take<Skip<Iter<T>>> {
        self.buf.iter().skip(row * self.col).take(self.col)
    }

    /// Returns the `col`nth column
    pub fn col(&self, col: usize) -> Take<StepBy<Skip<Iter<T>>>> {
        self.buf.iter().skip(col).step_by(self.col).take(self.row)
    }

    /// Returns the sum of two matrices
    /// ## Conditions
    /// - `self.dim() == other.dim()`
    pub fn add(&self, other: &Self) -> MatErr<Self> {
        if self.dim() != other.dim() {
            return Err(Add);
        }

        Ok(Self::from_map(self.dim(), |r, c| {
            self[(r, c)] + other[(r, c)]
        }))
    }

    /// Returns the subtraction of two matrices
    /// ## Conditions
    /// - `self.dim() == other.dim()`
    pub fn sub(&self, other: &Self) -> MatErr<Self> {
        if self.dim() != other.dim() {
            return Err(Add);
        }

        Ok(Self::from_map(self.dim(), |r, c| {
            self[(r, c)] - other[(r, c)]
        }))
    }

    /// Returns the hadamard product of two matrices, returning a
    /// matrix with dimensions `self.dim()`
    /// ## Conditions
    /// - `self.dim() == other.dim()`
    pub fn hadamard(&self, other: &Self) -> MatErr<Self> {
        if self.dim() != other.dim() {
            return Err(Mul);
        }

        Ok(Self::from_map(self.dim(), |r, c| {
            self[(r, c)] * other[(r, c)]
        }))
    }

    /// Returns the product of two matrices, returning a
    /// matrix with dimensions `(self.rows(), other.cols())`
    /// ## Conditions
    /// - `self.cols() == other.rows()`
    pub fn mul(&self, other: &Self) -> MatErr<Self> {
        if self.col != other.row {
            return Err(Mul);
        }

        Ok(Self::from_map((self.row, other.col), |r, c| {
            self
                .row(r)
                .zip(other.col(c))
                .fold(T::zero(), |acc, (v1, v2)| acc + (*v1) * (*v2))
        }))
    }

    /// Returns a matrix scaled by a factor of `scalar`
    pub fn scaled(&self, scalar: T) -> Self {
        Self::from_map(self.dim(), |r, c| scalar * self[(r, c)])
    }

    /// Returns a matrix transformed by `map`
    pub fn mapped<F>(&self, mut map: F) -> Self
    where
        F: FnMut(T) -> T
    {
        Self::from_map(self.dim(), |r, c| {
            map(self[(r, c)])
        })
    }
}

/// Iterator trait for collecting a matrix from an iterator
pub trait MatCollect<T: Num> 
where
    Self: Iterator + IntoIterator<Item = T>,
    Vec<T>: FromIterator<<Self as Iterator>::Item>
{
    fn to_matrix(&mut self, (row, col): (usize, usize)) -> MatErr<Mat<T>> {
        let buf = self.into_iter().collect();       
        Mat::<T>::from_vec((row, col), buf)
    }
}

impl<'m, T: Num, I> MatCollect<T> for I 
where 
    I: Iterator + IntoIterator<Item = T>,
    Vec<T>: FromIterator<<Self as Iterator>::Item>
{}

/// Rows matrix iterator 
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

/// Columns matrix iterator
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
