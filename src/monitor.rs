use eframe::egui;
use egui::pos2;
use std::sync::mpsc::Receiver;

use crate::Layer;

pub struct Monitor {
    rx: Receiver<(Option<Vec<Layer>>, Option<Layer>)>,
    _current_loss: f64,
    number_neurons: usize,
    hidden_layers: Option<Vec<Layer>>,
    output_layer: Option<Layer>,
}

impl Monitor {
    pub fn construct(
        rx: Receiver<(Option<Vec<Layer>>, Option<Layer>)>,
        hidden_neurons: usize,
    ) -> Self {
        Self {
            rx,
            _current_loss: 0.0,
            number_neurons: hidden_neurons,
            hidden_layers: None,
            output_layer: None,
        }
    }
}

impl eframe::App for Monitor {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        const NUMBER_INPUTS: usize = 2;
        const NUMBER_OUTPUTS: usize = 1;
        egui::CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();
            let rect = painter.clip_rect();

            let height = rect.height();
            let width = rect.width();

            let circle_stroke = egui::Stroke::new(3.0, egui::Color32::GREEN);
            // let debug_stroke = egui::Stroke::new(4.0, egui::Color32::RED);
            let fill = egui::Color32::TRANSPARENT;

            if let Ok(layers) = self.rx.try_recv() {
                (self.hidden_layers, self.output_layer) = layers;
            };
            if let (Some(hidden), Some(output)) = (&self.hidden_layers, &self.output_layer) {
                //Plus one for padding
                let total_columns = 1 + hidden.len() + 1;
                let neuron_layer_count = self.number_neurons;
                let x_step = width / (total_columns + 1) as f32;
                let y_input_step = height / (1 + NUMBER_INPUTS) as f32;
                let y_hidden_step = height / (1 + neuron_layer_count) as f32;
                let y_output_step = height / (1 + NUMBER_OUTPUTS) as f32;
                let radius = y_hidden_step / 5.0;

                let get_weight_stroke = |weight: f32| {
                    let normalized = ((weight + 10.0) / 20.0).clamp(0.0, 1.0);
                    let green_weight = (255.0 * normalized) as u8;
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(0, green_weight, 0))
                };

                if let Some(first_hidden) = hidden.first() {
                    for input_idx in 0..NUMBER_INPUTS {
                        for hidden_idx in 0..neuron_layer_count {
                            let weight = first_hidden.get_weights()[(input_idx, hidden_idx)] as f32;
                            let start = pos2(x_step * 1.0, (input_idx + 1) as f32 * y_input_step);
                            let end = pos2(x_step * 2.0, (hidden_idx + 1) as f32 * y_hidden_step);
                            painter.line_segment([start, end], get_weight_stroke(weight));
                        }
                    }
                }

                for l in 0..hidden.len().saturating_sub(1) {
                    let next_hidden_layer = &hidden[l + 1];
                    for current_idx in 0..neuron_layer_count {
                        for next_idx in 0..neuron_layer_count {
                            let weight =
                                next_hidden_layer.get_weights()[(current_idx, next_idx)] as f32;
                            let start = pos2(
                                x_step * (l + 2) as f32,
                                (current_idx + 1) as f32 * y_hidden_step,
                            );
                            let end = pos2(
                                x_step * (l + 3) as f32,
                                (next_idx + 1) as f32 * y_hidden_step,
                            );
                            painter.line_segment([start, end], get_weight_stroke(weight));
                        }
                    }
                }

                if hidden.last().is_some() {
                    let last_hidden_col_idx = hidden.len() + 1;
                    for hidden_idx in 0..neuron_layer_count {
                        for output_idx in 0..NUMBER_OUTPUTS {
                            let weight = output.get_weights()[(hidden_idx, output_idx)] as f32;
                            let start = pos2(
                                x_step * last_hidden_col_idx as f32,
                                (hidden_idx + 1) as f32 * y_hidden_step,
                            );
                            let end = pos2(
                                x_step * total_columns as f32,
                                (output_idx + 1) as f32 * y_output_step,
                            );
                            painter.line_segment([start, end], get_weight_stroke(weight));
                        }
                    }
                }

                for number_input in 1..=NUMBER_INPUTS {
                    painter.circle(
                        pos2(x_step, number_input as f32 * y_input_step),
                        radius,
                        fill,
                        circle_stroke,
                    ); //We use hidden step as the defining radius factor because we wan't the radius to be the same in every circle and the hidden y is usually the smallest
                }

                for layer in 1..=hidden.len() {
                    for neuron in 1..=neuron_layer_count {
                        painter.circle(
                            pos2(x_step * (layer + 1) as f32, neuron as f32 * y_hidden_step),
                            radius,
                            fill,
                            circle_stroke,
                        );
                    }
                }

                for number_output in 1..=NUMBER_OUTPUTS {
                    painter.circle(
                        pos2(
                            x_step * total_columns as f32,
                            number_output as f32 * y_output_step,
                        ),
                        radius,
                        fill,
                        circle_stroke,
                    );
                }
            }
        });
        ctx.request_repaint();
    }
}
