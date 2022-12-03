use std::marker::PhantomData;
use std::ops::Index;
use std::ops::IndexMut;
use serde_derive::Deserialize;
use serde_derive::Serialize;

use crate::linalg::LinAlg;
use crate::num::Num;

pub enum IndexType {
    Front(usize),
    Back(usize)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Array<M, N=f32> 
where
    N: Num,
    M: LinAlg<N>
{
    pub buf: Vec<M>,
    _t: PhantomData<N>
}

impl<M, N> Array<M, N> 
where
    N: Num,
    M: LinAlg<N>
{
    pub fn from_buf(buf: Vec<M>) -> Self {
        Self { 
            buf,
            _t: PhantomData::default()
        }
    }

    pub fn new() -> Self {
        Self { 
            buf: Vec::new(), 
            _t: PhantomData::default() 
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    fn to_index(&self, index: IndexType) -> usize {
        match index {
            IndexType::Back(i) => self.len() - (i + 1),
            IndexType::Front(i) => i
        }
    }

    pub fn indices_mut(&mut self, fst: IndexType, sec: IndexType) -> (&mut M, &mut M) {
        let fst = self.to_index(fst);
        let sec = self.to_index(sec);

        if fst == sec {
            panic!("tried to borrow index {} twice!", fst)
        }

        let min = fst.min(sec);
        let max = fst.max(sec);

        let (head, rest) = self.buf.split_at_mut(min+1);

        if fst == min {
            (&mut head[min], &mut rest[max-(min+1)])
        }
        else {
            (&mut rest[max-(min+1)], &mut head[min])
        }
    }
    
    pub fn zero(&mut self) -> &mut Self {
        for mat in self.buf.iter_mut() {
            mat.fill_eq(N::zero());
        }

        self
    }
}

impl<M, N> Index<usize> for Array<M, N>
where
    N: Num,
    M: LinAlg<N>
{
    type Output=M;

    fn index(&self, index: usize) -> &Self::Output {
        &self.buf[index]
    }
}

impl<M, N> IndexMut<usize> for Array<M, N>
where
    N: Num,
    M: LinAlg<N>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.buf[index]
    }
}

impl<M, N> Index<IndexType> for Array<M, N>
where
    N: Num,
    M: LinAlg<N>
{
    type Output=M;

    fn index(&self, index: IndexType) -> &Self::Output {
        match index {
            IndexType::Front(i) => &self.buf[i],
            IndexType::Back(i)  => &self.buf[self.len()-(i+1)],
        }
    }
}

impl<M, N> IndexMut<IndexType> for Array<M, N> 
where
    N: Num,
    M: LinAlg<N>
{
    fn index_mut(&mut self, index: IndexType) -> &mut Self::Output {
        let len = self.len();

        match index {
            IndexType::Front(i) => &mut self.buf[i],
            IndexType::Back(i)  => &mut self.buf[len-(i+1)],
        }
    }
}

