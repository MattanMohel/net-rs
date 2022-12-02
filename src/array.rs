use std::marker::PhantomData;
use std::ops::Index;
use std::ops::IndexMut;
use crate::linalg::LinAlg;
use crate::num::Num;

pub enum IndexType {
    Front(usize),
    Back(usize)
}

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
    
    pub fn fill(&mut self, fill: N) -> &mut Self {
        for mat in self.buf.iter_mut() {
            mat.fill_eq(fill);
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

