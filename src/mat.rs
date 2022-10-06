use std::ops::Add;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;
use std::fmt;

use super::num::Def;
use super::num::Num;

#[derive(Debug)]
pub enum MatErr {
    Mul,
    Add,
    Sqr,
    Inv
}

#[derive(Clone)]
pub struct Mat<N: Num = Def> {
    buf: Vec<N>,
    col: usize,
    row: usize,
}

impl<N: Num> Mat<N> {
    pub fn empty() -> Self {
        Self {
            buf: Vec::new(),
            col: 0,
            row: 0,
        }
    }

    pub fn from_arr<const C: usize, const R: usize>(arr: [[N; C]; R]) -> Self {
        let buf = arr.iter().flatten().cloned().collect();

        Self {
            buf,
            col: C,
            row: R,
        }
    }

    pub fn from_vec((col, row): (usize, usize), buf: Vec<N>) -> Self {
        Self { buf, col, row }
    }

    pub fn from_fn<F>((col, row): (usize, usize), f: F) -> Self
    where
        F: Fn() -> N,
    {
        let buf = (0..col * row).map(|_| f()).collect();

        Self { buf, col, row }
    }

    pub fn from_map<F>((col, row): (usize, usize), map: F) -> Self
    where
        F: Fn(usize, usize) -> N,
    {
        let buf = (0..col * row).map(|i| map(i / row, i % row)).collect();

        Self { buf, col, row }
    }

    pub fn fill((col, row): (usize, usize), fill: N) -> Self {
        Self::from_map((col, row), |_, _| fill)
    }

    pub fn zeros((col, row): (usize, usize)) -> Self {
        Self::from_map((col, row), |_, _| N::zero())
    }

    pub fn identity(len: usize) -> Self {
        Self::from_map((len, len), |c, r| if c == r { N::one() } else { N::zero() })
    }

    pub fn transpose(&self) -> Self {
        Self::from_map((self.row, self.col), |c, r| self[(r, c)])
    }

    pub fn cofactor(&self) -> Result<Self, MatErr> {
        if !self.is_square() {
            return Err(MatErr::Sqr);
        }

        Ok(Self::from_map(self.dim(), |c, r| {
            self.minor((c, r)).unwrap().determinant().unwrap()
                * if (c + r) % 2 == 0 { N::one()} else { N::neg() }
        }))
    }

    pub fn adjoint(&self) -> Result<Self, MatErr> {
        Ok(self.cofactor()?.transpose())
    }

    pub fn inverse(&self) -> Result<Self, MatErr> {
        let det = self.determinant()?;

        if det == N::zero() {
            return Err(MatErr::Inv)
        }
        
        Ok(self.adjoint()?.scale(det.inv()))
    }

    pub fn determinant(&self) -> Result<N, MatErr> {
        if !self.is_square() {
            return Err(MatErr::Sqr);
        }

        if self.col == 2 {
            return Ok(self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]);
        }

        Ok((0..self.row).fold(N::zero(), |acc, i| {
            self.minor((0, i)).unwrap().determinant().unwrap()
                * self[(0, i)]
                * if i % 2 == 0 { N::one() } else { N::neg() }
                + acc
        }))
    }

    pub fn minor(&self, (col, row): (usize, usize)) -> Result<Self, MatErr> {
        if !self.is_square() {
            return Err(MatErr::Sqr);
        }

        Ok(Self::from_map((self.col - 1, self.row - 1), |c, r| {
            let co = if c < col { 0 } else { 1 };
            let ro = if r < row { 0 } else { 1 };
            self[(c + co, r + ro)]
        }))
    }

    pub fn to_index((col, row): (usize, usize)) -> usize {
        col * row + row
    }

    pub fn to_coord(&self, index: usize) -> (usize, usize) {
        (index / self.row, index % self.row)
    }

    pub fn dim(&self) -> (usize, usize) {
        (self.col, self.row)
    }

    pub fn is_square(&self) -> bool {
        self.col == self.row
    }

    pub fn buf(&self) -> &Vec<N> {
        &self.buf
    }

    pub fn buf_mut(&mut self) -> &mut Vec<N> {
        &mut self.buf
    }

    pub fn cols(&self) -> Cols<'_, N> {
        Cols::from(self)
    }

    pub fn rows(&self) -> Rows<'_, N> {
        Rows::from(self)
    }

    pub fn iter(&self) -> MatIter<'_, N> {
        MatIter::from(self)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, MatErr> {
        if self.dim() != rhs.dim() {
            return Err(MatErr::Add);
        }

        Ok(Self::from_map(self.dim(), |c, r| {
            self[(c, r)] + rhs[(c, r)]
        }))
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, MatErr> {
        if self.row != rhs.col {
            return Err(MatErr::Mul);
        }

        Ok(Self::from_map((self.col, rhs.row), |c, r| {
            self.rows()
                .get(c)
                .unwrap()
                .iter()
                .zip(rhs.cols().get(r).unwrap().iter())
                .fold(N::zero(), |acc, (v1, v2)| acc + (*v1) * (*v2))
        }))
    }

    pub fn scale(&self, scalar: N) -> Self {
        Self::from_map(self.dim(), |c, r| scalar * self[(c, r)])
    }
}

impl<N: Num> Index<(usize, usize)> for Mat<N> {
    type Output = N;

    fn index(&self, (col, row): (usize, usize)) -> &Self::Output {
        &self.buf[self.row * col + row]
    }
}

impl<N: Num> fmt::Display for Mat<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let buf = self.rows().fold(String::new(), |acc, row| format!("{}\n{:?}", acc, row));
        write!(f, "{}", buf)
    }
}

pub struct MatIter<'m, N: Num> {
    mat: &'m Mat<N>,
    i: usize,
}

pub struct Cols<'m, N: Num> {
    mat: &'m Mat<N>,
    i: usize,
}

impl<'m, N: Num> From<&'m Mat<N>> for MatIter<'m, N> {
    fn from(mat: &'m Mat<N>) -> Self {
        MatIter { mat, i: 0 }
    }
}

impl<'m, N: Num> Iterator for MatIter<'m, N> {
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        Some(self.mat.buf()[self.i - 1])
    }
}

impl<'m, N: Num> Cols<'m, N> {
    fn get(&self, col: usize) -> Option<Vec<N>> {
        if col >= self.mat.dim().1 {
            return None;
        }

        Some(
            self.mat
                .buf()
                .iter()
                .enumerate()
                .filter_map(|(i, v)| {
                    if i % self.mat.dim().1 == col {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .collect(),
        )
    }
}

impl<'m, N: Num> From<&'m Mat<N>> for Cols<'m, N> {
    fn from(mat: &'m Mat<N>) -> Self {
        Cols { mat, i: 0 }
    }
}

impl<'m, N: Num> Iterator for Cols<'m, N> {
    type Item = Vec<N>;

    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        self.get(self.i - 1)
    }
}

pub struct Rows<'m, N: Num> {
    mat: &'m Mat<N>,
    i: usize,
}

impl<'m, N: Num> Rows<'m, N> {
    fn get(&self, row: usize) -> Option<Vec<N>> {
        if row >= self.mat.dim().0 {
            return None;
        }

        Some(
            self.mat
                .buf()
                .iter()
                .enumerate()
                .filter_map(|(i, v)| {
                    if i / self.mat.dim().1 == row {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .collect(),
        )
    }
}

impl<'m, N: Num> From<&'m Mat<N>> for Rows<'m, N> {
    fn from(mat: &'m Mat<N>) -> Self {
        Rows { mat, i: 0 }
    }
}

impl<'m, N: Num> Iterator for Rows<'m, N> {
    type Item = Vec<N>;

    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        self.get(self.i - 1)
    }
}