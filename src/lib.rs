use ndarray::{Array1};
use ndarray_rand::{RandomExt, rand_distr::{Uniform}};

pub struct Neuron {
    weights: Array1<f64>,
    bias: f64
}

impl Neuron {
    pub fn construct(n_inputs: usize) -> Self {
        Neuron { 
            weights: (Array1::random(n_inputs, Uniform::new(0.0, 1.0).unwrap())), 
            bias: (rand::random_range(-1.0..1.0))
        }
    }
}

pub fn forward(neuron: &Neuron, input: Array1<f64>) -> f64 {
    let scalar_sum = input.dot(&neuron.weights);
    let z = scalar_sum + neuron.bias;
    1.0 / (1.0 + (-z).exp())
}