# Neural Network Demo

A small project I built to actually understand how neural networks work under the hood, not just use them through a library.

The idea is pretty simple: train a tiny network to solve XOR from scratch, and watch it learn in real time through a GUI. No PyTorch, no TensorFlow, just hand-rolled backprop in Rust.

## What it does

- Implements a feedforward neural network with configurable hidden layers and neurons
- Trains on the XOR problem (two binary inputs, one binary output)
- Uses sigmoid activations and gradient descent with learning rate decay
- Shows the network's weights and activations live while it trains, via an egui window

## Why I built this

I got tired of treating neural networks as black boxes. I wanted to feel what backpropagation actually does: how the error signal flows backwards, how weights nudge themselves in the right direction, and why things like learning rate decay matter.

There are definitely rough edges here. The backpropagation is a simplified version and the architecture is not exactly state-of-the-art. But it learns XOR, and watching it converge is genuinely satisfying.
<img width="927" height="1036" alt="image" src="https://github.com/user-attachments/assets/fc73aea0-7cdc-4805-a0d0-422bae018df5" />
