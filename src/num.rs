use std::fmt;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Neg;
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
    Div<Output=Self> +
    Neg<Output=Self> +
    MulAssign + 
    AddAssign + 
    SubAssign +
    Copy + Sized + fmt::Display + fmt::Debug
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
