use std::ops::Index;
use super::num::*;

/// type representing matrix dimensions
type Dim   = (usize, usize);
/// type representing matrix coordinates
type Coord = (usize, usize);
/// type representing a matrix slice's index transformation
type Trans = fn(Coord, Dim) -> Coord;

/// NOTE!!!\
/// its impossible to have an 'impl T' return in a 
/// trait body so this system has to be reworked
/// 
/// Rather, Matrix will be a single type (with no trait implementations)
/// 
/// To create a regular Matrix, `Matrix<T, Vec<T>>` will be used
/// 
/// Top create a slice, `Matrix<T, &Vec<T>>` will be used
/// 
/// To get element from a matrix the same concept of 'indexing through a closure' will 
/// be used with the same concept of closure composition. On the other hand, since there 
/// will be no traits, the closures will actually be initializable using 'return impl Closure' notation


pub trait MatrixType<T: Num> 
where
    Self: Index<Coord, Output=T>
{
    fn dim(&self) -> Dim;
    fn row(&self) -> usize;
    fn col(&self) -> usize;
    fn buf(&self) -> &Vec<T>;

    
    fn slice<F>(&self, dim: Dim, trans: F) -> MatrixSlice<'_, T, F>
    where 
        F: Fn((Coord, Dim)) -> (Coord, Dim);


    fn is_square(&self) -> bool {
        self.row() == self.col()
    }

    fn to_index(&self, (r, c): Coord) -> usize {
        self.col() * r + c
    }

    // fn row_slice<F>(&self, r: usize) -> MatrixSlice<'_, T, F> 
    // where
    //     F: Fn((Coord, Dim)) -> (Coord, Dim)
    // {
    //     self.slice((1, self.col()), |(i, j), _| (i, j))
    // }

    // fn transpose<F>(&self) -> MatrixSlice<'_, T, F> 
    // where
    //     F: Fn((Coord, Dim)) -> (Coord, Dim)    
    // {
    //     self.slice((self.col(), self.row()), |(i, j), _| (j, i))
    // }

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

    fn slice<F>(&self, (row, col): Dim, trans: F) -> MatrixSlice<'_, T, F> 
    where
        F: Fn((Coord, Dim)) -> (Coord, Dim)
    {
        MatrixSlice { 
            buf: self.buf(), 
            row: row, 
            col: col, 
            comp: trans
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

pub struct MatrixSlice<'src, T, F> 
where
    T: Num,
    F: Fn((Coord, Dim)) -> (Dim, Coord)
{
    buf: &'src Vec<T>,
    row: usize,
    col: usize,
    comp: F,
}

impl<'src, T, F> MatrixSlice<'src, T, F> 
where
    T: Num,
    F: Fn((Coord, Dim)) -> (Coord, Dim)
{
    fn test(&self, (row, col): Dim, f: F) -> MatrixSlice<'_, T, impl Fn((Coord, Dim)) -> (Coord, Dim)> {
        MatrixSlice { 
            buf: &self.buf, 
            row: 0, 
            col: 0, 
            comp: f
        }
    }

    
    fn s<G>(&self, (row, col): Dim, f: G) -> MatrixSlice<'_, T, impl Fn((Coord, Dim)) -> (Coord, Dim)>
    where
        G: Fn((Coord, Dim)) -> (Coord, Dim)
    {
        // let f = |i: (Coord, Dim)| (self.comp)(i);

        MatrixSlice { 
            buf: self.buf(), 
            row: row, 
            col: col,
            comp: |((i, j), (r, c))| ((i, j), (r, c))
        }
    }
}

impl<'src, T, F> MatrixType<T> for MatrixSlice<'src, T, F> 
where
    T: Num,
    F: Fn((Coord, Dim)) -> (Dim, Coord)
{
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

    fn slice<G>(&self, (row, col): Dim, f: G) -> MatrixSlice<'_, T, G>
    where
        G: Fn((Coord, Dim)) -> (Coord, Dim)
    {
        // let f = |i: (Coord, Dim)| (self.comp)(i);

        // MatrixSlice { 
        //     buf: self.buf(), 
        //     row: row, 
        //     col: col,
        //     comp: |((i, j), (r, c))| ((i, j), (r, c))
        // }

        todo!()
    }
}

impl<'src, T, F> Index<Coord> for MatrixSlice<'src, T, F>
where
    T: Num,
    F: Fn((Coord, Dim)) -> (Coord, Dim)  
{
    type Output=T;

    fn index(&self, coord: Coord) -> &Self::Output {
        // let trans = 
        //     self.comp
        //         .iter()
        //         .zip(self.dims.iter())
        //         .fold(coord, |a, (trans, dim)| trans(a, *dim));

        // &self.buf[self.to_index(trans)]

        todo!()
    }
}