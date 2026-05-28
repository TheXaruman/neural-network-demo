use std::sync::mpsc::Receiver;
use eframe::egui;

pub struct Monitor {
    rx: Receiver<f64>,
    current_loss: f64,
}

impl Monitor {
    fn construct(rx: Receiver<f64>) -> Self {
        Self {
            rx,
            current_loss: 0.0,
        }
    }
}

impl eframe::App for Monitor {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        
    }
}