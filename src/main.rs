use ndarray::{Array1, array, s};
use neuralnetwork::*;
use std::sync::mpsc;
use std::thread;

const HIDDEN_LAYER_NEURONS: usize = 32;
const HIDDEN_LAYER_COUNT: usize = 5;
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
        let mut layers: Vec<Layer> = {
            let mut l = Vec::with_capacity(HIDDEN_LAYER_COUNT);
            l.push(Layer::construct(2, HIDDEN_LAYER_NEURONS));
            for _ in 1..HIDDEN_LAYER_COUNT {
                l.push(Layer::construct(HIDDEN_LAYER_NEURONS, HIDDEN_LAYER_NEURONS));
            }
            l
        };
        let mut decision_neutron = Layer::construct(HIDDEN_LAYER_NEURONS, 1);
        for epoch in 1..=10000 {
            let mut _loss = 0.0;
            for _ in 1..=100 {
                let training = rand::random_range(0..4) as usize;
                let input_slice = set.slice(s![training, 0..2]).to_owned();
                let target = set[[training, 2]];
                let mut hidden_output: Vec<Array1<f64>> =
                    vec![Array1::zeros(HIDDEN_LAYER_NEURONS); HIDDEN_LAYER_COUNT];
                hidden_output[0] = forward_layer(&layers[0], &input_slice);
                for layer_number in 1..HIDDEN_LAYER_COUNT {
                    let prev_output = hidden_output[layer_number - 1].clone();
                    hidden_output[layer_number] =
                        forward_layer(&layers[layer_number], &prev_output);
                }
                let decision_output =
                    forward_f64(&decision_neutron, &hidden_output[HIDDEN_LAYER_COUNT - 1]);
                let error_signal: f64 = decision_output - target;
                decision_neutron.train(
                    array![decision_output],
                    hidden_output[HIDDEN_LAYER_COUNT - 1].clone(),
                    array![error_signal],
                );

                let mut hidden_error: Array1<f64> = Array1::zeros(HIDDEN_LAYER_NEURONS);
                for (layer_number, layer) in layers.iter_mut().enumerate() {
                    for i in 0..layer.get_bias().len() {
                        hidden_error[i] = error_signal * decision_neutron.get_weights()[(i, 0)];
                    }
                    let layer_input = if layer_number == 0 {
                        input_slice.clone()
                    } else {
                        hidden_output[layer_number - 1].clone()
                    };
                    layer.train(
                        hidden_output[layer_number].clone(),
                        layer_input,
                        hidden_error.clone(),
                    );
                }
                _loss += error_signal.powi(2);
            }
            if epoch % 20 == 0 {
                Layer::decay_learning_rate(0.9999);
            }

            tx.send((Some(layers.clone()), Some(decision_neutron.clone())))
                .ok();
        }
    });
    // for loss in rx {
    //     println!("{}", loss);
    // }
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Neural Network Demo",
        options,
        Box::new(|_cc| Ok(Box::new(Monitor::construct(rx, HIDDEN_LAYER_NEURONS)))),
    )
    .unwrap();
}
