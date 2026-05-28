use ndarray::{array, s};
use neuralnetwork::*;
fn main() {
    //first two numbers are the input the last number the target
    let set = array![
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let mut layer = Layer::construct(2, 3);
    for _ in 1..100 {
        let mut loss = 0.0;
        for _ in 1..100 {
            let training = rand::random_range(0..4) as usize;
            let input_slice = set.slice(s![training, 0..2]).to_owned();
            let target = set[[training, 2]];
            let output = forward(&layer, &input_slice);
            let error_signal: f64 = output - target;
            layer.train(output, input_slice, error_signal);

            loss = loss + error_signal.powi(2);
        }
        Neuron::decay_learning_rate(0.99);
        println!("{}", loss / 100.0);
    }
}
