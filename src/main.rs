use matrix::Matrix;
use matrix::IMatrix;
use net::Network;
use num::Num;

pub mod matrix;
pub mod num;
pub mod net;
pub mod cost;
pub mod num_fn;
pub mod activation;
pub mod mnist;
pub mod one_hot;
pub mod matrix_slice;

fn print_matrix<T: Num>(m: &Matrix<T>) {
    println!("\ndim: {:?}\n", m.dim());
    for i in 0..m.row() {
        for j in 0..m.col() {
            print!("{} ", m[(i, j)]);
        }
        println!()
    }
}

fn main() {
    let _network = Network::new([3, 5, 3]);

    let m1 = Matrix::<i32>::from_arr([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
    let m2 = m1.transpose();
    let prod = m1.mul(&m2);
    
    print_matrix(&m1);
    print_matrix(&m2);
    print_matrix(&prod);

    println!("det of m1: {}", prod.determinant());
}