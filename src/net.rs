use super::num_fn::NumeralFn;
use super::activation::Activation;
use super::cost::Cost;
use super::num::N;
use super::mat::*;

const LEARN_RATE: f64 = 0.01;
const BATCH_SIZE: usize = 64;

pub struct Meta<const L: usize> {
    // layer sizes
    pub form: [usize; L],
    // size of batch sampling
    pub batch_size: usize,
    // learning coefficient
    pub learn_rate: f64,
    // activation function
    pub activation: Activation,
    // cost function
    pub cost: Cost,
}

impl<const L: usize> Meta<L> {
    pub fn new(form: [usize; L]) -> Self {
        Self { 
            form,
            batch_size: BATCH_SIZE, 
            learn_rate: LEARN_RATE, 
            activation: Activation::Sig, 
            cost: Cost::Quad, 
        }
    }
}

pub struct Network<const L: usize> {
    // hyper parameters
    meta_data: Meta<L>,
    // weight matrices layers
    weights: Vec<Matrix>,
    // bias matrices layers
    biases: Vec<Matrix>,

    /////////////////////
    /// Training Data ///
    /////////////////////

    // activations buffer
    activations: Vec<Matrix>,
    // layer sums buffer
    sums: Vec<Matrix>,
    // layer error buffer
    errors: Vec<Matrix>,
    // weight error buffer
    weight_errors: Vec<Matrix>
}

impl<const L: usize> Network<L> {
    pub fn new(form: [usize; L]) -> Self {
        if form.len() <= 2 {
            panic!()
        }

        Self { 
            meta_data: Meta::new(form),
            
            biases: {
                (0..form.len())
                    .map(|i| Matrix::zeros((form[i], 1)))
                    .collect()
            },
            
            weights: {
                (0..form.len()-1)
                    .map(|i| Matrix::random((form[i], form[i+1])))
                    .collect()
            },

            activations: {
                (0..form.len())
                    .map(|i| Matrix::zeros((form[i], 1)))
                    .collect()
            },

            sums: {
                (1..form.len())
                    .map(|i| Matrix::zeros((form[i], 1)))
                    .collect()
            },

            errors: {
                (0..form.len()-1)
                    .map(|i| Matrix::zeros((form[i], 1)))
                    .collect()
            },

            weight_errors: {
                (0..form.len()-1)
                    .map(|i| Matrix::zeros((form[i], form[i+1])))
                    .collect()
            }
        }
    }

    /// Clears training buffers
    pub fn clear_train_data(&mut self) {
        self.activations.clear();
        self.sums.clear();
        self.errors.clear();
        self.weight_errors.clear();
    }

    /// Applies activation function to one number
    fn activate(&self, n: N) -> N {
        self.meta_data.activation.value(n)
    }

    /// Applies activation derivative to one number
    fn d_activate(&self, n: N) -> N {
        self.meta_data.activation.deriv(n)
    }

    /// Computes layer cost matrix
    fn cost_matrix(&self, exp: &Matrix,  layer: usize) -> Matrix {
        Matrix::from_map(exp.dim(), |r_c| {
            self.meta_data.cost.value((exp[r_c], self.activations[layer][r_c]))
        })
    }

    /// Computes layer cost derivative matrix
    fn d_cost_matrix(&self, exp: &Matrix,  layer: usize) -> Matrix {
        Matrix::from_map(exp.dim(), |r_c| {
            self.meta_data.cost.deriv((exp[r_c], self.activations[layer][r_c]))
        })
    }

    pub fn forward_prop(&self, input: &Matrix<N>) -> Matrix {
        if input.row() != self.meta_data.form[0] {
            panic!()
        }

        self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(input.clone(), |acc, (weight, bias)| {
                weight
                    .mul(&acc)
                    .add(&bias)
                    .map(|n| self.activate(n))
            })
    }

    pub fn backward_prop(&mut self, input: Vec<&Matrix>, expected: Vec<&Matrix>) {
        if input.len() != expected.len() {
            panic!()
        }

        // for k in 0.. {
        //     let mut bias_error: Vec<_> = self.meta_data.form
        //         .iter()
        //         .skip(1)
        //         .map(|l| Matrix::zeros((*l, 1)))
        //         .collect();

        //     let mut weight_error: Vec<_> = self
        //         .form()
        //         .zip(self.form().skip(1))
        //         .map(|(l1, l2)| Matrix::zeros((*l2, *l1)))
        //         .collect();

        //     for i in k*self.meta_data.batch_size..(k+1)*self.meta_data.batch_size {
        //         let (bias_error_l, weight_error_l) = self.train(input[i], expected[i]);
                
        //         for j in 0..bias_error.len() {
        //             bias_error[j].add_eq(&bias_error_l[j].scale(self.learn_rate/self.batch_size as f64));
        //             weight_error[j].add_eq(&weight_error_l[j].scale(self.learn_rate/self.batch_size as f64));
        //         }
        //     }
        // }




        // let epoch_steps = (expected.len() as f32 / (self.batch_size) as f32).ceil() as usize;

        // ISSUE: batch learn rate not always learn rate as last batch will not be exactly
        // equal to the batch size (data count not always divisible by batch_size)

        // TODO: move error vectors into Net member so it doesn't have to
        // reallocate buffers for every trait - same for Vec<weights, sums, errors, weight_errors>

        // for _ in 0..TRAIN_EPOCH {
        //     let mut error_acc = Vec::new();    
        //     let mut weight_error_acc = Vec::new();    

        //     for i in 0..input.len() {
        //         if i % self.batch_size == 0 {
        //             // reset accumulator
        //             (error_acc, weight_error_acc) = self.train(input[0], expected[0]);

        //             for (j, l) in (0..L-1).map(|i| (i, L-2-i)) {   
        //                 // apply errors
        //                 self.weights[i].add_eq(&weight_error_acc[l]);
        //                 self.biases[i].add_eq(&error_acc[l]);
        //             }
        //         }

        //         let (error, weight_error) = self.train(input[i], expected[i]);

        //         for j in 0..L-1 {
        //             error_acc[j].add_eq(&error[j].scale(self.learn_scale()));
        //             weight_error_acc[j].add_eq(&weight_error[j].scale(self.learn_scale()));
        //         }
        //     }
        // }
    }

    pub fn train(&mut self, input: &Matrix, expected: &Matrix) {
        // clear previous training data
        self.clear_train_data();
        
        // push initial input as layer_0
        self.activations.push(input.clone());

        for i in 0..L-1 {
            // sum_l = weights_l * activations_l-1 + biases_l
            let sum_l = self.weights[i]
                .mul(&self.activations[i])
                .add(&self.biases[i]);
            // activation_l = activate( sum_l )
            let activation_l = sum_l
                .map(|n| self.activate(n));

            self.activations.push(activation_l);
            self.sums.push(sum_l);
        }

        // error_L = d_activation( sum_L-1 ) * cost( activations_L )
        let error = self.sums[L-2]
            .map(|n| self.d_activate(n))
            .diagonal()
            .mul(&self.cost_matrix(&expected, L-1));

        self.errors.push(error);

        for (i, l) in (0..L-1).map(|i| (i, L-2-i)) {
            // weight_error_L = error_L * transpose( activations_L-1 )
            let weight_error_l = self.errors[L-2-l]
                .mul(&self.activations[l].transpose());

            self.weight_errors.push(weight_error_l);

            if l == 0 {
                break
            }

            // error_L-1 = diagonal . d_activation( sum_L-1 ) * transpose( weights_L ) * error_L
            let error_l = self.sums[l-1]
                .map(|n| self.d_activate(n))
                .diagonal()
                .mul(&self.weights[l].transpose())
                .mul(&self.errors[i]);

            self.errors.push(error_l);
        }
    }
}