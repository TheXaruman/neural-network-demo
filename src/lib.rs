use ndarray::{Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use std::sync::{LazyLock, Mutex};

static LEARNING_RATE: LazyLock<Mutex<f64>> = LazyLock::new(|| Mutex::new(0.1));
#[derive(Clone)]
pub struct Layer {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

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
    //use ndarray::array;
    #[test]
    fn construct_random_weights() {
        let layer_1 = Layer::construct(3, 3);
        let layer_2 = Layer::construct(3, 3);

        assert_ne!(layer_1.weights, layer_2.weights)
    }

    #[test]
    fn construct_random_bias() {
        let layer_1 = Layer::construct(3, 3);
        let layer_2 = Layer::construct(3, 3);

        assert_ne!(layer_1.bias, layer_2.bias)
    }

    //#[test]
    //#[should_panic(expected = "Vectors need to have the same length for dot product")]
    //fn forward_panic() {
    //  let test_neuron = Layer::construct(5, 3);
    //  let inputs = array!(1.0, 3.0);
    //  forward(&test_neuron, &inputs);
    //}
    //#[test]
    //fn vector_addition() {
    //    let test_neuron = Layer::construct(2, 5);
    //    let inputs = array!(1.0, 3.0);
    //    //let result = forward(&test_neuron, &inputs);
    //    assert!(result != array![0.0]);
    //}

    //#[test]
    //fn learning_rate_decay() {
    //    let learning_rate_before = *LEARNING_RATE.lock().unwrap();
    //    Layer::decay_learning_rate(0.99);
    //    let learning_rate_after = *LEARNING_RATE.lock().unwrap();
    //    assert!(learning_rate_before > learning_rate_after);
    //}
}
