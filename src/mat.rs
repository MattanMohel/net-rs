use std::iter::{
    Skip,
    Take,
    StepBy
};
use std::process::Output;
use std::slice::Iter;
use super::num::Num;
use super::num::N; 
use std::ops::Index;
use rand::prelude::*;
use rand::distributions::Standard;

type Row<'m, T: Num> = Take<Skip<Iter<'m, T>>>;
type Col<'m, T: Num> = Take<StepBy<Skip<Iter<'m, T>>>>;

// make matrix trait implementing all these, have one 'specialized' function
// called 'index' that will retrive the index given an 'i' value - row/col stride/elem-count
// Implement for one Matrix type with original buffer and one Matrix type with burrowed buffer
// i.e. transpose won't create a new buffer but will 'transform' the strides of a burrowed matrix
// cloning a burrowed matrix will create a new matrix with a non-borrowed buffer

#[derive(Clone)]
pub struct Mat<T: Num> {
    buf: Vec<T>,
    row: usize,
    col: usize,
}

impl<T: Num> Mat<T> {
    pub fn from_arr<const C: usize, const R: usize>(arr: [[T; C]; R]) -> Self {
        let buf = arr.iter().copied().flatten().collect();

        Self {
            buf,
            col: C,
            row: R,
        }
    }

    pub fn from_fn<F>((row, col): (usize, usize), mut map: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let buf = (0..row*col).map(|i| map(i / col, i % col)).collect();

        Self { 
            buf, 
            row, 
            col,
        }
    }

    pub fn identity(len: usize) -> Self {
        Self::from_fn((len, len), |r, c| {
            if c == r { 
                T::one() 
            } 
            else { 
                T::zero() 
            }
        })
    }
    
    pub fn fill((row, col): (usize, usize), fill: T) -> Self {
        Self::from_fn((row, col), |_, _| fill)
    }

    pub fn zeros((row, col): (usize, usize)) -> Self {
        Self::from_fn((row, col), |_, _| T::zero())
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
        Self::from_fn((row, col), |_, _| rng.gen() * (max - min) + min)
    }

    pub fn diagonal(&self) -> Self {
        if self.row > 1 && self.col > 1 {
            panic!()
        }

        if self.row == 1 {
            Self::from_fn((self.col, self.col), |r, c| {
                if c == r {
                    self[(0, c)] 
                } 
                else {
                    T::zero()
                }
            })
        } 
        else {
            Self::from_fn((self.row, self.row), |r, c| {
                if c == r { 
                    self[(r, 0)] 
                } 
                else {
                    T::zero()
                }
            })
        }
    }

    pub fn transpose(&self) -> Self {
        Self::from_fn((self.col, self.row), |r, c| self[(c, r)])
    }

    pub fn cofactor(&self) -> Self {
        if !self.is_square() {
            panic!("took cofactor of non-square matrix")
        }

        Self::from_fn(self.dim(), |r, c| {
            self.minor((r, c))
                * if (r + c) % 2 == 0 { T::one()} else { -T::one() }
        })
    }

    pub fn inverse(&self) -> Self {
        let det = self.determinant();

        if det == T::zero() {
            panic!("took inverse of matrix with determinant of 0")
        }
        
        self
            .cofactor()
            .transpose()
            .scaled(T::one() / det)
    }

    pub fn determinant(&self) -> T {
        if !self.is_square() {
            panic!("took determinant of non-square matrix")
        }

        if self.col == 2 {
            return self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)];
        }

        (0..self.row).fold(T::zero(), |acc, i| {
            self.minor((0, i))
                * self[(0, i)]
                * if i % 2 == 0 { T::one() } else { -T::one() }
                + acc
        })
    }

    pub fn minor(&self, (row, col): (usize, usize)) -> T {
        if !self.is_square() {
            panic!("took minor of non-square matrix")
        }

        Self::from_fn((self.row - 1, self.col - 1), |r, c| {
            let ro = if r < row { 0 } else { 1 };
            let co = if c < col { 0 } else { 1 };
            self[(r + ro, c + co)]
        })
        .determinant()
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
    
    pub fn row(&self, row: usize) -> Row<'_, T> {
        self.buf.iter().skip(row * self.col).take(self.col)
    }

    pub fn col(&self, col: usize) -> Col<'_, T> {
        self.buf.iter().skip(col).step_by(self.col).take(self.row)
    }

    pub fn add(&self, other: &Self) -> Self {
        if self.dim() != other.dim() {
            panic!("addition dimension mismatch")
        }

        Self::from_fn(self.dim(), |r, c| {
            self[(r, c)] + other[(r, c)]
        })
    }

    pub fn sub(&self, other: &Self) -> Self {
        if self.dim() != other.dim() {
            panic!("subtraction dimension mismatch")
        }

        Self::from_fn(self.dim(), |r, c| {
            self[(r, c)] - other[(r, c)]
        })
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        if self.col != rhs.row {
            panic!("multipication dimension mismatch")
        }

        Self::from_fn((self.row, rhs.col), |r, c| {
            self
                .row(r)
                .zip(rhs.col(c))
                .fold(T::zero(), |acc, (v1, v2)| acc + *v1 * *v2)
        })
    }

    pub fn scaled(&self, scalar: T) -> Self {
        Self::from_fn(self.dim(), |r, c| scalar * self[(r, c)])
    }
}

impl<T: Num> Index<(usize, usize)> for Mat<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.buf[self.col * row + col]
    }
}
