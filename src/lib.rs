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

pub fn forward(neuron: &Neuron, input: &Array1<f64>) -> f64 {
    assert_eq!(neuron.weights.len(), input.len(), "Vectors need to have the same length for dot product");
    let scalar_sum = input.dot(&neuron.weights);
    let z = scalar_sum + neuron.bias;
    1.0 / (1.0 + (-z).exp())
}

#[cfg(test)]
mod test{
    use ndarray::array;
    use super::*;
    #[test]
    fn construct_random_weights() {
        let neuron_1 = Neuron::construct(3);
        let neuron_2 = Neuron::construct(3);

        assert_ne!(neuron_1.weights, neuron_2.weights)
    }

    #[test]
    fn construct_random_bias() {
        let neuron_1 = Neuron::construct(3);
        let neuron_2 = Neuron::construct(3);

        assert_ne!(neuron_1.bias, neuron_2.bias)
    }

    #[test]
    #[should_panic(expected = "Vectors need to have the same length for dot product")]
    fn vector_length_difference_panic() {
        let neuron = Neuron::construct(5);
        let inputs = array!(1.0, 3.0);
        forward(&neuron, &inputs);
    }
}