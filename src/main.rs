use ndarray::{Array1, array, s};
use neuralnetwork::*;
use std::sync::mpsc;
use std::thread;

const HIDDEN_LAYER_NEURONS: usize = 64;
fn main() {
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        //first two numbers are the input the last number the target
        let set = array![
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ];
        let mut layer = Layer::construct(2, HIDDEN_LAYER_NEURONS);
        let mut decision_neutron = Layer::construct(HIDDEN_LAYER_NEURONS, 1);
        for epoch in 1..=10000 {
            let mut loss = 0.0;
            for _ in 1..=100 {
                let training = rand::random_range(0..4) as usize;
                let input_slice = set.slice(s![training, 0..2]).to_owned();
                let target = set[[training, 2]];
                let hidden_output = forward_layer(&layer, &input_slice);
                let decision_output = forward_f64(&decision_neutron, &hidden_output);
                let error_signal: f64 = decision_output - target;
                decision_neutron.train(
                    array![decision_output],
                    hidden_output.clone(),
                    array![error_signal],
                );
                let mut hidden_error: Array1<f64> = Array1::zeros(HIDDEN_LAYER_NEURONS);
                for i in 0..layer.get_bias().len() {
                    hidden_error[i] = error_signal * decision_neutron.get_weights()[(i, 0)];
                }
                layer.train(hidden_output, input_slice, hidden_error);
                loss += error_signal.powi(2);
            }
            if epoch % 100 == 0 {
                Layer::decay_learning_rate(0.9999);
            }
            let average_loss = loss / 100.0;
            tx.send(average_loss).ok();

        }
    });
    for loss in rx {
        println!("{}", loss);
    }
    // let options = eframe::NativeOptions::default();
    // eframe::run_native("Neural Network Demo", options, Box::new(|_cc| Ok(Box::new(Monitor::construct(rx, HIDDEN_LAYER_NEURONS))))).unwrap();
}
