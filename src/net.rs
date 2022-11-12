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
    // layer error accumulation buffer
    error_acc: Vec<Matrix>,
    // weight error buffer
    weight_errors: Vec<Matrix>,
    // weight error accumulation buffer
    weight_error_acc: Vec<Matrix>
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

            error_acc: {
                (0..form.len()-1)
                    .map(|i| Matrix::zeros((form[i], 1)))
                    .collect()
            },

            weight_errors: {
                (0..form.len()-1)
                    .map(|i| Matrix::zeros((form[i], form[i+1])))
                    .collect()
            },

            weight_error_acc: {
                (0..form.len()-1)
                    .map(|i| Matrix::zeros((form[i], form[i+1])))
                    .collect()
            }
        }
    }

    /// Clears propagation buffers
    pub fn clear_propagation_data(&mut self) {
        self.activations.clear();
        self.sums.clear();
        self.errors.clear();
        self.weight_errors.clear();
    }

    /// Clears accumulation buffers
    pub fn clear_accumulation_data(&mut self) {
        self.error_acc.clear();
        self.weight_error_acc.clear()
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

    pub fn apply_gradient(&mut self, sample_size: usize) {
        // coefficient of learn rate
        let learn_rate = self.meta_data.learn_rate / sample_size as f64;

        // apply stochastic error gradient 
        for j in 0..L-1 {
            self.biases[j].add_eq(&self.error_acc[j].scale(learn_rate));
            self.weights[j].add_eq(&self.weight_error_acc[j].scale(learn_rate));
        }
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

    pub fn backward_prop(&mut self, inputs: &Vec<Matrix>, expected: &Vec<Matrix>) {
        if inputs.len() != expected.len() {
            panic!()
        }

        self.clear_accumulation_data();

        for (i, (input, expected)) in inputs.iter().zip(expected.iter()).enumerate() {
            self.train(input, expected);
            
            // accumulate errors
            for j in 0..L-1 {
                self.error_acc[j].add_eq(&self.errors[j]);
                self.weight_error_acc[j].add_eq(&self.weight_errors[j]);
            }

            if i % self.meta_data.batch_size == 0 {
                self.apply_gradient(self.meta_data.batch_size);
                self.clear_accumulation_data();
            }

            let remainder = inputs.len() % self.meta_data.batch_size;
            self.apply_gradient(remainder);
        }
    }

    // NOTE: d_cost not used? check equation sheet for reference 

    /// Trains network against a provided set of inputs and expected outputs,  
    /// storing the error results in the 'errors' and 'weight_errors' buffers
    pub fn train(&mut self, input: &Matrix, expected: &Matrix) {
        // clear previous training data
        self.clear_propagation_data();
        
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
            .mul(&self.d_cost_matrix(&expected, L-1));

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