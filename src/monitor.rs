use std::sync::mpsc::Receiver;
use eframe::egui;
use egui::{Key::W, Pos2, Vec2, pos2};

pub struct Monitor {
    rx: Receiver<f64>,
    current_loss: f64,
    number_neurons: usize,
}

impl Monitor {
    pub fn construct(rx: Receiver<f64>, hidden_neurons: usize) -> Self {
        Self {
            rx,
            current_loss: 0.0,
            number_neurons: hidden_neurons,
        }
    }
}

impl eframe::App for Monitor {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        while let Ok(new_loss) = self.rx.try_recv() {
            self.current_loss = new_loss;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            
            let painter = ui.painter();
            let rect = painter.clip_rect();

            let height = rect.height();
            let width = rect.width();
            
            let row_unit = height / 10.0;
            let column_unit = width / 10.0;
            let radius = height / 10.0 / (self.number_neurons as f32 / 2.5);

            let stroke = egui::Stroke::new(4.0, egui::Color32::GREEN);
            // let debug_stroke = egui::Stroke::new(4.0, egui::Color32::RED);
            let fill = egui::Color32::TRANSPARENT;

            // painter.line_segment([pos2(0.0, 0.0), pos2(width, 0.0)], stroke);
            // painter.line_segment([pos2(0.0, row_unit * 1.0), pos2(width, row_unit * 1.0)], stroke);
            // painter.line_segment([pos2(0.0, row_unit * 2.0), pos2(width, row_unit * 2.0)], stroke);
            // painter.line_segment([pos2(0.0, row_unit * 3.0), pos2(width, row_unit * 3.0)], stroke);
            // painter.line_segment([pos2(0.0, row_unit * 4.0), pos2(width, row_unit * 4.0)], stroke);
            // painter.line_segment([pos2(0.0, row_unit * 5.0), pos2(width, row_unit * 5.0)], stroke);
            // painter.line_segment([pos2(0.0, row_unit * 6.0), pos2(width, row_unit * 6.0)], stroke);
            // painter.line_segment([pos2(0.0, row_unit * 7.0), pos2(width, row_unit * 7.0)], stroke);
            // painter.line_segment([pos2(0.0, row_unit * 8.0), pos2(width, row_unit * 8.0)], stroke);
            // painter.line_segment([pos2(0.0, row_unit * 9.0), pos2(width, row_unit * 9.0)], stroke);
            // painter.line_segment([pos2(0.0, row_unit * 10.0), pos2(width, row_unit * 10.0)], stroke);


            // painter.line_segment([pos2(0.0, 0.0), pos2(0.0, height)], stroke);
            // painter.line_segment([pos2(column_unit * 1.0, 0.0), pos2(column_unit * 1.0, height)], stroke);
            // painter.line_segment([pos2(column_unit * 2.0, 0.0), pos2(column_unit * 2.0, height)], stroke);
            // painter.line_segment([pos2(column_unit * 3.0, 0.0), pos2(column_unit * 3.0, height)], stroke);
            // painter.line_segment([pos2(column_unit * 4.0, 0.0), pos2(column_unit * 4.0, height)], stroke);
            // painter.line_segment([pos2(column_unit * 5.0, 0.0), pos2(column_unit * 5.0, height)], stroke);
            // painter.line_segment([pos2(column_unit * 6.0, 0.0), pos2(column_unit * 6.0, height)], stroke);
            // painter.line_segment([pos2(column_unit * 7.0, 0.0), pos2(column_unit * 7.0, height)], stroke);
            // painter.line_segment([pos2(column_unit * 8.0, 0.0), pos2(column_unit * 8.0, height)], stroke);
            // painter.line_segment([pos2(column_unit * 9.0, 0.0), pos2(column_unit * 9.0, height)], stroke);
            // painter.line_segment([pos2(column_unit * 10.0, 0.0), pos2(column_unit * 10.0, height)], stroke);

            painter.circle(
                egui::pos2(column_unit * 2.0, row_unit * 3.0), 
                radius, 
                fill, 
                stroke
            );
            
            painter.circle(
                egui::pos2(column_unit * 2.0, row_unit * 7.0), 
                radius, 
                fill, 
                stroke
            );

            let y_pos_unit = (height - 2.0 * row_unit) / self.number_neurons as f32;



            for i in 0..self.number_neurons {
                painter.circle(
                egui::pos2(column_unit * 5.0, y_pos_unit * (i+1) as f32), 
                radius, 
                fill, 
                stroke
            );
            }
        });
        ctx.request_repaint();
    }
}