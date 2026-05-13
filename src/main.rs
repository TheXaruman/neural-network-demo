use neuralnetwork::*;
use ndarray::array;
fn main() {
    let neuron = Neuron::construct(3);
    let input = array![1.0, 2.0, 3.0];
    let result = forward(&neuron, &input);
    println!("{}", result);
}
