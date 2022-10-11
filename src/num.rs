use std::fmt;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::Div;
use std::ops::Mul;
use std::ops::MulAssign;
use std::*;
use std::ops::SubAssign;

use rand::distributions::Distribution;

pub type N = f64;

pub trait Num:
    PartialEq +
    Add<Output=Self> + 
    Sub<Output=Self> +
    Mul<Output=Self> +
    MulAssign + 
    AddAssign + 
    SubAssign +
    Copy + Sized + fmt::Display + fmt::Debug
{
    fn inv(&self) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn neg() -> Self;
}

impl Num for i32 {
    fn inv(&self) -> Self {
        1_i32 / *self
    }
    fn zero() -> Self {
        0_i32
    }
    fn one() -> Self {
        1_i32
    }
    fn neg() -> Self {
        -1_i32
    }
}

impl Num for i64 {
    fn inv(&self) -> Self {
        1_i64 / *self
    }
    fn zero() -> Self {
        0_i64
    }
    fn one() -> Self {
        1_i64
    }
    fn neg() -> Self {
        -1_i64
    }
}

impl Num for f32 {
    fn inv(&self) -> Self {
        1_f32 / *self
    }
    fn zero() -> Self {
        0_f32
    }
    fn one() -> Self {
        1_f32
    }
    fn neg() -> Self {
        -1_f32
    }
}

impl Num for f64 {
    fn inv(&self) -> Self {
        1_f64 / *self
    }
    fn zero() -> Self {
        0_f64
    }
    fn one() -> Self {
        1_f64
    }
    fn neg() -> Self {
        -1_f64
    }
}
