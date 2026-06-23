use eframe::egui;
use egui::pos2;
use std::sync::mpsc::Receiver;

use crate::Layer;

pub struct Monitor {
    rx: Receiver<(Option<Layer>, Option<Layer>)>,
    _current_loss: f64,
    number_neurons: usize,
    hidden_layer: Option<Layer>,
    output_layer: Option<Layer>,
}

impl Monitor {
    pub fn construct(rx: Receiver<(Option<Layer>, Option<Layer>)>, hidden_neurons: usize) -> Self {
        Self {
            rx,
            _current_loss: 0.0,
            number_neurons: hidden_neurons,
            hidden_layer: None,
            output_layer: None,
        }
    }
}

impl eframe::App for Monitor {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();
            let rect = painter.clip_rect();

            let height = rect.height();
            let width = rect.width();

            let row_unit = height / 10.0;
            let column_unit = width / 10.0;
            let radius = height / 10.0 / (self.number_neurons as f32 / 2.5);

            let circle_stroke = egui::Stroke::new(4.0, egui::Color32::GREEN);
            // let debug_stroke = egui::Stroke::new(4.0, egui::Color32::RED);
            let fill = egui::Color32::TRANSPARENT;

            (self.hidden_layer, self.output_layer) = match self.rx.try_recv() {
                Ok(layers) => layers,
                Err(_) => return,
            };

            painter.circle(
                pos2(column_unit * 2.0, row_unit * 3.0),
                radius,
                fill,
                circle_stroke,
            );

            painter.circle(
                egui::pos2(column_unit * 2.0, row_unit * 7.0),
                radius,
                fill,
                circle_stroke,
            );

            let y_pos_unit = height / (self.number_neurons as f32 + 1.0);

            for i in 0..self.number_neurons {
                painter.circle(
                    egui::pos2(column_unit * 5.0, y_pos_unit * (i + 1) as f32),
                    radius,
                    fill,
                    circle_stroke,
                );
            }

            painter.circle(
                pos2(column_unit * 8.0, height / 2.0),
                radius,
                fill,
                circle_stroke,
            );

            for u in 0..2 {
                for i in 0..self.number_neurons {
                    let branch_weight =
                        (self.hidden_layer.as_ref().unwrap().get_weights()[(u, i)] + 10.0) / 20.0;
                    let green_weight = (255.0 * branch_weight) as u8;
                    let weight_stroke =
                        egui::Stroke::new(4.0, egui::Color32::from_rgb(0, green_weight, 0));
                    let start_point = match u {
                        0 => pos2(column_unit * 2.0, row_unit * 3.0),
                        1 => pos2(column_unit * 2.0, row_unit * 7.0),
                        _ => unreachable!(),
                    };
                    let end_point = pos2(column_unit * 5.0, y_pos_unit * (i + 1) as f32);
                    painter.line_segment([start_point, end_point], weight_stroke);
                }
            }
            for u in 0..1 {
                for i in 0..self.number_neurons {
                    let branch_weight =
                        (self.output_layer.as_ref().unwrap().get_weights()[(i, u)] + 10.0) / 20.0;
                    let green_weight = (255.0 * branch_weight) as u8;
                    let weight_stroke =
                        egui::Stroke::new(4.0, egui::Color32::from_rgb(0, green_weight, 0));
                    let start_point = pos2(column_unit * 5.0, y_pos_unit * (i + 1) as f32);
                    let end_point = pos2(column_unit * 8.0, height / 2.0);
                    painter.line_segment([start_point, end_point], weight_stroke);

                }
            }
        });
        ctx.request_repaint();
    }
}
