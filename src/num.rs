use std::*;
use std::fmt;
use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::MulAssign;

pub type Def = f32;

pub trait Num: 
    Add<Output=Self> + 
    Sub<Output=Self> + 
    Mul<Output=Self> + 
    MulAssign + Copy + 
    fmt::Display + Sized
{
    fn zero() -> Self;
    fn one() -> Self;
}

impl Num for i32 {
    fn zero() -> Self {
        0_i32
    }
    fn one() -> Self {
        1_i32
    }
}

impl Num for i64 {
    fn zero() -> Self {
        0_i64
    }
    fn one() -> Self {
        1_i64
    }
}

impl Num for u32 {
    fn zero() -> Self {
        0_u32
    }
    fn one() -> Self {
        1_u32
    }
}

impl Num for u64 {
    fn zero() -> Self {
        0_u64
    }
    fn one() -> Self {
        1_u64
    }
}

impl Num for usize {
    fn zero() -> Self {
        0_usize
    }
    fn one() -> Self {
        1_usize
    }
}

impl Num for f32 {
    fn zero() -> Self {
        0_f32
    }
    fn one() -> Self {
        1_f32
    }
}

impl Num for f64 {
    fn zero() -> Self {
        0_f64
    }
    fn one() -> Self {
        1_f64
    }
}