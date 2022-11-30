
use super::step::Step;
use super::cost::Cost;
use super::linalg::*;

type N = f32;

const LEARN_RATE: N = 0.03;
const BATCH_SIZE: usize = 64;

pub struct Meta<const L: usize> {
    // layer sizes
    pub form: [usize; L],
    // size of batch sampling
    pub batch_size: usize,
    // learning coefficient
    pub learn_rate: N,
    // step function
    pub step: Step,
    // cost function
    pub cost: Cost,
}

impl<const L: usize> Meta<L> {
    pub fn new(form: [usize; L]) -> Self {
        Self { 
            form,
            batch_size: BATCH_SIZE, 
            learn_rate: LEARN_RATE, 
            step: Step::Sig, 
            cost: Cost::Quad, 
        }
    }
}


/// NOTE: NETWORK FOR DEBUG PURPOSES - REVERSE ENGINEER BUG!!!
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

    // steps buffer
    steps: Vec<Matrix>,
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
                (1..form.len())
                    .map(|i| Matrix::from_zeros((form[i], 1)))
                    .collect()
            },
            
            weights: {
                (0..form.len()-1)
                    .map(|i| Matrix::rand((form[i+1], form[i])))
                    .collect()
            },

            steps: Vec::new(),

            sums: Vec::new(),

            errors: Vec::new(),

            weight_errors: Vec::new(),

            error_acc: {
                (1..form.len())
                    .map(|i| Matrix::from_zeros((form[i], 1)))
                    .collect()
            },


            weight_error_acc: {
                (0..form.len()-1)
                    .map(|i| Matrix::from_zeros((form[i+1], form[i])))
                    .collect()
            }
        }
    }

    /// Clears propagation buffers
    pub fn clear_propagation_data(&mut self) {
        self.steps.clear();
        self.sums.clear();
        self.errors.clear();
        self.weight_errors.clear();
    }

    /// Clears accumulation buffers
    pub fn clear_accumulation_data(&mut self) {
        for err in self.error_acc.iter_mut() {
            err.fill_eq(0.);
        }
        for weight_err in  self.weight_error_acc.iter_mut() {
            weight_err.fill_eq(0.);
        }
    }

    /// Applies step function to one number
    fn activate(&self, n: N) -> N {
        self.meta_data.step.value(n)
    }

    /// Applies step derivative to one number
    fn d_activate(&self, n: N) -> N {
        self.meta_data.step.deriv(n)
    }

    /// Computes layer cost matrix
    fn cost_matrix<M: LinAlg>(&self, exp: &M,  layer: usize) -> Matrix {
        Matrix::from_map(exp.shape(), |r_c| {
            self.meta_data.cost.value(exp[M::to_dim(r_c)] - self.steps[layer][r_c])
        })
    }

    /// Computes layer cost derivative matrix
    fn d_cost_matrix<M: LinAlg>(&self, exp: &M,  layer: usize) -> Matrix {
        Matrix::from_map(exp.shape(), |r_c| {
            self.meta_data.cost.deriv(exp[M::to_dim(r_c)] - self.steps[layer][r_c])
        })
    }

    pub fn apply_gradient(&mut self, sample_size: usize) {
        // coefficient of learn rate
        let learn_rate = self.meta_data.learn_rate / sample_size as N;

        // apply stochastic error gradient 
        for j in 0..L-1 {
            self.biases[j].add_eq(&self.error_acc[j].scale(learn_rate));
            self.weights[j].add_eq(&self.weight_error_acc[j].scale(learn_rate));
        }
    }

    pub fn forward_prop(&self, input: &Vector) -> Matrix {
        if input.row() != self.meta_data.form[0] {
            panic!()
        }

        self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(input.to_matrix(), |acc, (weight, bias)| {
                weight
                    .mul::<_, Matrix>(&acc)
                    .add(bias)
                    .map(|n| self.activate(n))
            })
    }

    /// Trains network against a provided set of inputs and expected outputs,  
    /// storing the error results in the 'errors' and 'weight_errors' buffers
    pub fn back_prop(&mut self, input: &Vector, expected: &Vector) {
        // clear previous training data
        self.clear_propagation_data();
        
        // push initial input as layer_0
        self.steps.push(input.to_matrix());

        for i in 0..L-1 {
            // sum_l = weights_l * steps_l-1 + biases_l
            let sum_l = self.weights[i]
                .mul::<_, Matrix>(&self.steps[i])
                .add(&self.biases[i]);
            // step_l = activate( sum_l )
            let step_l = sum_l
                .map(|n| self.activate(n));

            self.steps.push(step_l);
            self.sums.push(sum_l);
        }

        // error_L = d_step( sum_L-1 ) * cost( steps_L )
        let error = self.sums[L-2]
            .map(|n| self.d_activate(n))
            .to_vector()
            .diagonal()
            .mul(&self.d_cost_matrix(expected, L-1));

        self.errors.push(error);

        for (i, l) in (0..L-1).map(|i| (i, L-2-i)) {
            // weight_error_L = error_L * transpose( steps_L-1 )
            let weight_error_l = self.errors[L-2-l]
                .mul(&self.steps[l].transpose());

            self.weight_errors.push(weight_error_l);

            if l == 0 {
                break
            }

            // error_L-1 = diagonal . d_step( sum_L-1 ) * transpose( weights_L ) * error_L
            let error_l = self.sums[l-1]
                .map(|n| self.d_activate(n))
                .to_vector()
                .diagonal()
                .mul::<_, Matrix>(&self.weights[l].transpose())
                .mul(&self.errors[i]);

            self.errors.push(error_l);
        }

        // reverses layer error orders to ascending
        self.weight_errors.reverse();
        self.errors.reverse();
    }

    pub fn train(&mut self, inputs: &[Vector], expected: &[Vector], take: usize) {
        if inputs.len() != expected.len() {
            panic!()
        }

        self.clear_accumulation_data();

        for (i, (input, expected)) in inputs.iter().take(take).zip(expected.iter()).enumerate() {
            self.back_prop(input, expected);
            
            // accumulate errors
            for j in 0..self.errors.len() {
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
}