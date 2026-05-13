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
    let mut neuron = Neuron::construct(2);
    let mut loss = 0.0;
    for _ in 1..10 {
        for _ in 1..1000 {
            let training = rand::random_range(0..4) as usize;
            let input_slice = set.slice(s![training, 0..2]).to_owned();
            let target = set[[training, 2]];
            let output = forward(&neuron, &input_slice);
            let error_signal: f64 = output - target;
            neuron.train(output, input_slice, error_signal);
            loss = (loss + error_signal) / 2.0;
        }
        println!("{}", loss);
    }
}
