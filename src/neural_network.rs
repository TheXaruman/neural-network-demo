use ndarray::{Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use serde::{Deserialize, Serialize};
use std::sync::{LazyLock, Mutex};

#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

static LEARNING_RATE: LazyLock<Mutex<f64>> = LazyLock::new(|| Mutex::new(0.1));

impl Layer {
    pub fn construct(n_inputs: usize, n_neurons: usize) -> Self {
        let weight_distribution = match Uniform::new(-1.0, 1.0) {
            Ok(dist) => dist,
            Err(err) => panic!("paniced with: {}", err),
        };
        let bias_distribution = match Uniform::new(-1.0, 1.0) {
            Ok(dist) => dist,
            Err(err) => panic!("paniced with: {}", err),
        };
        Layer {
            weights: Array2::random((n_inputs, n_neurons), weight_distribution),
            bias: Array1::random(n_neurons, bias_distribution),
        }
    }

    pub fn train(&mut self, output: Array1<f64>, inputs: Array1<f64>, error_signal: Array1<f64>) {
        assert_eq!(
            self.weights.nrows(),
            inputs.len(),
            "Vectors need to have the same length for dot product"
        );
        let current_learning_rate = LEARNING_RATE.lock().unwrap();
        for neuron in 0..self.bias.len() {
            let sigmoid_gradient = output[neuron] * (1.0 - output[neuron]);
            for (weight, input) in &mut self
                .weights
                .column_mut(neuron)
                .iter_mut()
                .zip(inputs.iter())
            {
                let delta_weight =
                    *current_learning_rate * error_signal[neuron] * sigmoid_gradient * input;
                *weight -= delta_weight;
            }
            self.bias[neuron] -= *current_learning_rate * error_signal[neuron] * sigmoid_gradient;
        }
    }

    pub fn decay_learning_rate(decaying_factor: f64) {
        let mut learning_rate = LEARNING_RATE.lock().unwrap();
        *learning_rate *= decaying_factor;
    }

    pub fn get_weights(&self) -> Array2<f64> {
        self.weights.clone()
    }

    pub fn get_bias(&self) -> Array1<f64> {
        self.bias.clone()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    pub layers: Vec<Layer>,
    pub decision: Layer,
}

use std::fs::File;
use std::io::{BufReader, BufWriter};

impl Model {
    pub fn save(&self, path: &str) {
        let file = File::create(path).expect("cannot create file");
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self).expect("save failed");
    }

    pub fn load(path: &str) -> Self {
        let file = File::open(path).expect("cannot open file");
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).expect("load failed")
    }
}

pub fn forward_layer(layer: &Layer, input: &Array1<f64>) -> Array1<f64> {
    let scalar_sum = input.dot(&layer.weights);
    let z = scalar_sum + &layer.bias;
    z.mapv(|zvalue| 1.0 / (1.0 + (-zvalue).exp()))
}

pub fn forward_f64(neuron: &Layer, input: &Array1<f64>) -> f64 {
    let one_d_weights: Array1<f64> = neuron
        .weights
        .clone()
        .into_shape_with_order(neuron.weights.nrows() * neuron.weights.ncols())
        .unwrap();
    assert_eq!(neuron.bias.len(), 1);
    let zero_d_bias: f64 = neuron.bias[0];
    assert_eq!(
        one_d_weights.len(),
        input.len(),
        "Vectors need to have the same length for dot product"
    );
    let scalar_sum = input.dot(&one_d_weights);
    let z = scalar_sum + zero_d_bias;

    1.0 / (1.0 + (-z).exp())
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;
    #[test]
    fn construct_random_weights() {
        let layer_1 = Layer::construct(3, 3);
        let layer_2 = Layer::construct(3, 3);
        assert_ne!(layer_1.weights, layer_2.weights);
    }

    #[test]
    fn construct_random_bias() {
        let layer_1 = Layer::construct(3, 3);
        let layer_2 = Layer::construct(3, 3);
        assert_ne!(layer_1.bias, layer_2.bias);
    }

    #[test]
    fn construct_weight_shape() {
        let layer = Layer::construct(4, 7);
        let w = layer.get_weights();
        assert_eq!(w.nrows(), 4);
        assert_eq!(w.ncols(), 7);
    }

    #[test]
    fn construct_bias_length() {
        let layer = Layer::construct(4, 7);
        assert_eq!(layer.get_bias().len(), 7);
    }

    #[test]
    fn construct_weights_in_range() {
        // Weights are drawn from Uniform(-1, 1).
        let layer = Layer::construct(10, 10);
        for &w in layer.get_weights().iter() {
            assert!((-1.0..1.0).contains(&w), "weight {w} outside (-1, 1)");
        }
    }

    #[test]
    fn construct_bias_in_range() {
        // Bias values are also drawn from Uniform(-1, 1).
        let layer = Layer::construct(10, 10);
        for &b in layer.get_bias().iter() {
            assert!((-1.0..1.0).contains(&b), "bias {b} outside (-1, 1)");
        }
    }

    #[test]
    fn forward_layer_output_length() {
        let layer = Layer::construct(3, 5);
        let input = array![0.5, -0.3, 1.0];
        let output = forward_layer(&layer, &input);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn forward_layer_sigmoid_range() {
        let layer = Layer::construct(3, 5);
        let input = array![100.0, -100.0, 0.0];
        let output = forward_layer(&layer, &input);
        for &val in output.iter() {
            assert!(
                (0.0..=1.0).contains(&val),
                "sigmoid output {val} outside [0, 1]"
            );
        }
    }

    #[test]
    fn forward_layer_known_values() {
        let layer = Layer::construct(2, 4);
        let out_a = forward_layer(&layer, &array![1.0, 0.0]);
        let out_b = forward_layer(&layer, &array![0.0, 1.0]);
        // For a randomly initialised layer these should virtually always differ.
        assert_ne!(out_a, out_b);
    }

    #[test]
    fn forward_f64_output_range() {
        let neuron = Layer::construct(4, 1);
        let input = array![0.1, -0.2, 0.3, -0.4];
        let output = forward_f64(&neuron, &input);
        assert!(
            output > 0.0 && output < 1.0,
            "sigmoid output {output} outside (0, 1)"
        );
    }

    #[test]
    #[should_panic(expected = "Vectors need to have the same length for dot product")]
    fn forward_f64_panics_on_mismatched_input() {
        let neuron = Layer::construct(5, 1);
        let short_input = array![1.0, 2.0];
        forward_f64(&neuron, &short_input);
    }

    #[test]
    #[should_panic]
    fn forward_f64_panics_when_not_single_output() {
        let neuron = Layer::construct(3, 3);
        let input = array![1.0, 2.0, 3.0];
        forward_f64(&neuron, &input);
    }

    #[test]
    fn train_updates_weights() {
        let mut layer = Layer::construct(2, 3);
        let weights_before = layer.get_weights();

        let output = array![0.6, 0.4, 0.7];
        let inputs = array![1.0, -1.0];
        let error = array![0.5, -0.3, 0.8];
        layer.train(output, inputs, error);

        assert_ne!(
            layer.get_weights(),
            weights_before,
            "weights should change after a train step with non-zero error"
        );
    }

    #[test]
    fn train_updates_bias() {
        let mut layer = Layer::construct(2, 3);
        let bias_before = layer.get_bias();

        let output = array![0.6, 0.4, 0.7];
        let inputs = array![1.0, -1.0];
        let error = array![0.5, -0.3, 0.8];
        layer.train(output, inputs, error);

        assert_ne!(
            layer.get_bias(),
            bias_before,
            "bias should change after a train step with non-zero error"
        );
    }

    #[test]
    fn train_zero_error_leaves_params_unchanged() {
        let mut layer = Layer::construct(2, 3);
        let weights_before = layer.get_weights();
        let bias_before = layer.get_bias();

        let output = array![0.6, 0.4, 0.7];
        let inputs = array![1.0, -1.0];
        let error = array![0.0, 0.0, 0.0];
        layer.train(output, inputs, error);

        assert_eq!(layer.get_weights(), weights_before);
        assert_eq!(layer.get_bias(), bias_before);
    }

    #[test]
    #[should_panic(expected = "Vectors need to have the same length for dot product")]
    fn train_panics_on_wrong_input_length() {
        let mut layer = Layer::construct(3, 2);
        let output = array![0.5, 0.5];
        let inputs = array![1.0];
        let error = array![0.1, 0.1];
        layer.train(output, inputs, error);
    }

    #[test]
    fn get_weights_returns_clone() {
        let layer = Layer::construct(3, 4);
        let w1 = layer.get_weights();
        let w2 = layer.get_weights();
        assert_eq!(w1, w2);
    }

    #[test]
    fn get_bias_returns_clone() {
        let layer = Layer::construct(3, 4);
        let b1 = layer.get_bias();
        let b2 = layer.get_bias();
        assert_eq!(b1, b2);
    }

    #[test]
    fn decay_learning_rate_decreases_rate() {
        let before = *LEARNING_RATE.lock().unwrap();
        assert!(before > 0.0, "learning rate must be positive before decay");

        Layer::decay_learning_rate(0.5);

        let lr_after = *LEARNING_RATE.lock().unwrap();
        assert!(
            lr_after > 0.0,
            "learning rate must remain positive after decay"
        );
    }

    #[test]
    fn decay_learning_rate_by_one_is_identity() {
        let lr_before = *LEARNING_RATE.lock().unwrap();

        Layer::decay_learning_rate(1.0);

        let lr_after = *LEARNING_RATE.lock().unwrap();
        assert!(
            (lr_after - lr_before).abs() < f64::EPSILON,
            "multiplying by 1.0 should be a no-op, got {lr_before} -> {lr_after}"
        );
    }
}
