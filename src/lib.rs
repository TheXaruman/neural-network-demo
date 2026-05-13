use ndarray::Array1;
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use std::sync::{LazyLock, Mutex};

static LEARNING_RATE: LazyLock<Mutex<f64>> = LazyLock::new(|| Mutex::new(0.1));

pub struct Neuron {
    weights: Array1<f64>,
    bias: f64,
}

impl Neuron {
    pub fn construct(n_inputs: usize) -> Self {
        let distribution = match Uniform::new(0.0, 1.0) {
      Ok(dist) => dist,
      Err(err) => panic!("paniced with: {}", err)           
    };
        Neuron {
            weights: Array1::random(n_inputs, distribution),
            bias: (rand::random_range(-1.0..1.0)),
        }
    }

    pub fn train(&mut self, output: f64, inputs: Array1<f64>, error_signal: f64) {
        assert_eq!(
            self.weights.len(),
            inputs.len(),
            "Vectors need to have the same length for dot product"
        );
        let sigmoid_gradient = output * (1.0 - output);
        let current_learning_rate = LEARNING_RATE.lock().unwrap();
        for (weight, input) in &mut self.weights.iter_mut().zip(inputs.iter()) {
            let delta_weight = *current_learning_rate * error_signal * sigmoid_gradient * input;
            *weight -= delta_weight;
        }
        self.bias -= *current_learning_rate * error_signal * sigmoid_gradient;
    }

    pub fn decay_learning_rate(decaying_factor: f64) {
        let mut learning_rate = LEARNING_RATE.lock().unwrap();
        *learning_rate *= decaying_factor;
    }
}

pub fn forward(neuron: &Neuron, input: &Array1<f64>) -> f64 {
    assert_eq!(
        neuron.weights.len(),
        input.len(),
        "Vectors need to have the same length for dot product"
    );
    let scalar_sum = input.dot(&neuron.weights);
    let z = scalar_sum + neuron.bias;
    1.0 / (1.0 + (-z).exp())
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;
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
        let test_neuron = Neuron::construct(5);
        let inputs = array!(1.0, 3.0);
        forward(&test_neuron, &inputs);
    }
    #[test]
    fn vector_addition() {
        let test_neuron = Neuron::construct(2);
        let inputs = array!(1.0, 3.0);
        let result = forward(&test_neuron, &inputs);
        assert!(result != 0.0);
    }

        #[test]
    fn learning_rate_decay() {
        let learning_rate_before = *LEARNING_RATE.lock().unwrap();
        Neuron::decay_learning_rate(0.99);
        let learning_rate_after = *LEARNING_RATE.lock().unwrap();
        assert!(learning_rate_before > learning_rate_after);
    }    
}
