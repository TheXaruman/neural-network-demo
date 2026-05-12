use std::f64::consts::E;
use ndarray::{Array1};
use ndarray_rand::{RandomExt, rand_distr::{Uniform}};
use rand;

struct Neuron {
    weights: Array1<f64>,
    bias: f64
}

impl Neuron {
    fn construct() -> Self {
        let n_inputs = 3;
        Neuron { 
            weights: (Array1::random(n_inputs, Uniform::new(0.0, 1.0).unwrap())), 
            bias: (rand::random_range(-1.0..1.0))
        }
    }
    
}