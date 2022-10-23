use std::ops::Index;
use super::num::*;

/// type representing matrix dimensions
type Dim   = (usize, usize);
/// type representing matrix coordinates
type Coord = (usize, usize);
/// type representing a matrix slice's index transformation
type Trans = fn(Coord, Dim) -> Coord;

pub trait MatrixType<T: Num> 
where
    Self: Index<Coord, Output=T>
{
    fn dim(&self) -> Dim;
    fn row(&self) -> usize;
    fn col(&self) -> usize;
    fn buf(&self) -> &Vec<T>;
    fn slice(&self, dim: Dim, f: Trans) -> MatrixSlice<'_, T>;

    fn is_square(&self) -> bool {
        self.row() == self.col()
    }

    fn to_index(&self, (r, c): Coord) -> usize {
        self.col() * r + c
    }

    fn transpose(&self) -> MatrixSlice<'_, T> {
        self.slice((self.col(), self.row()), |(i, j), _| (j, i))
    }

    fn add(&self, other: &Self) -> Matrix<T> {
        if self.dim() != other.dim() {
            // TODO!
            panic!()
        }

        Matrix::<T>::from_map(self.dim(), |r, c| {
            self[(r, c)] + other[(r, c)]
        })
    }

    fn sub(&self, other: &Self) -> Matrix<T> {
        if self.dim() != other.dim() {
            // TODO!
            panic!()
        }

        Matrix::<T>::from_map(self.dim(), |r, c| {
            self[(r, c)] - other[(r, c)]
        })
    }
}

pub struct Matrix<T: Num> {
    buf: Vec<T>,
    row: usize,
    col: usize,
}

impl<T: Num> MatrixType<T> for Matrix<T> {
    fn dim(&self) -> Dim {
        (self.row, self.col)
    }

    fn row(&self) -> usize {
        self.row
    }

    fn col(&self) -> usize {
        self.col
    }

    fn buf(&self) -> &Vec<T> {
        &self.buf
    }

    fn slice(&self, (row, col): Dim, f: Trans) -> MatrixSlice<'_, T> {
        MatrixSlice { 
            buf: self.buf(), 
            row: row, 
            col: col, 
            comp: vec![f],
            dims: vec![(row, col)]
        }
    }
}

impl<T: Num> Index<Coord> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): Coord) -> &Self::Output {
        &self.buf[self.col * row + col]
    }
}

impl<T: Num> Matrix<T> {
    pub fn from_arr<const C: usize, const R: usize>(arr: [[T; C]; R]) -> Self {
        let buf = 
            arr.iter()
               .copied()
               .flatten()
               .collect();

        Self {
            buf,
            col: C,
            row: R,
        }
    }

    pub fn from_map<F>((row, col): Dim, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let buf = 
            (0..row*col).map(|i| f(i / col, i % col))
                        .collect();

        Self { 
            buf, 
            row, 
            col,
        }
    }

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

    pub fn fill((row, col): Dim, fill: T) -> Self {
        Self::from_map((row, col), |_, _| fill)
    }

    pub fn zeros((row, col): Dim) -> Self {
        Self::from_map((row, col), |_, _| T::zero())
    }

    pub fn empty() -> Self {
        Self {
            buf: Vec::new(),
            col: 0,
            row: 0,
        }
    }
}

pub struct MatrixSlice<'src, T: Num> {
    buf: &'src Vec<T>,
    row: usize,
    col: usize,
    comp: Vec<Trans>,
    dims: Vec<Dim>
}

impl<'src, T: Num> MatrixType<T> for MatrixSlice<'src, T> {
    fn dim(&self) -> Dim {
        (self.row, self.col)
    }

    fn row(&self) -> usize {
        self.row
    }

    fn col(&self) -> usize {
        self.col
    }

    fn buf(&self) -> &Vec<T> {
        &self.buf
    }

    fn slice(&self, (row, col): Dim, f: Trans) -> MatrixSlice<'_, T> {
        let mut comp = self.comp.clone();
        comp.push(f);

        let mut dims = self.dims.clone();
        dims.push((row, col));

        MatrixSlice { 
            buf: self.buf(), 
            row: row, 
            col: col,
            comp,
            dims
        }
    }
}

impl<'src, T: Num> Index<Coord> for MatrixSlice<'src, T> {
    type Output=T;

    fn index(&self, coord: Coord) -> &Self::Output {
        let trans = 
            self.comp
                .iter()
                .zip(self.dims.iter())
                .fold(coord, |a, (trans, dim)| trans(a, *dim));

        &self.buf[self.to_index(trans)]
    }
}