use std::ops::Add;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;

use super::num::Def;
use super::num::Num;

#[derive(Debug)]
pub enum MatErr {
    Mul,
    Add,
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

    pub fn dim(&self) -> (usize, usize) {
        (self.col, self.row)
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
}

impl<N: Num> Index<(usize, usize)> for Mat<N> {
    type Output = N;

    fn index(&self, (col, row): (usize, usize)) -> &Self::Output {
        &self.buf[self.row * col + row]
    }
}

pub struct Cols<'m, N: Num> {
    mat: &'m Mat<N>,
    i: usize,
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

impl<N: Num> Mul for &Mat<N> {
    type Output = Result<Mat<N>, MatErr>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.row != rhs.col {
            return Err(MatErr::Mul);
        }

        Ok(Mat::<N>::from_map((self.col, rhs.row), |c, r| {
            self.rows()
                .get(c)
                .unwrap()
                .iter()
                .zip(rhs.cols().get(r).unwrap().iter())
                .fold(N::zero(), |acc, (v1, v2)| acc + (*v1) * (*v2))
        }))
    }
}

impl<N: Num> Add for &Mat<N> {
    type Output = Result<Mat<N>, MatErr>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.dim() != rhs.dim() {
            return Err(MatErr::Add);
        }

        Ok(Mat::<N>::from_map(self.dim(), |c, r| {
            self[(c, r)] + rhs[(c, r)]
        }))
    }
}

impl<N: Num> Mul<N> for &Mat<N> {
    type Output = Mat<N>;

    fn mul(self, rhs: N) -> Self::Output {
        Mat::<N>::from_map(self.dim(), |c, r| rhs * self[(c, r)])
    }
}

impl<N: Num> MulAssign<N> for Mat<N> {
    fn mul_assign(&mut self, rhs: N) {
        for v in self.buf_mut().iter_mut() {
            *v *= rhs;
        }
    }
}
