use num::Num;

pub struct Vector<T: Num> {
    buf: Vec<T>
    row: usize,
}

