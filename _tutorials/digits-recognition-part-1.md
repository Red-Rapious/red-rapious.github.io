---
title: "Handwritten digits recognition from scratch in Rust - Part 1: The Neural Netowrk"
excerpt: "The first step of the series: we'll implement the neural network."
permalink: /digits-recognition-part-0/
toc: true
toc_label: "The Neural Network"
tags:
    - deep learning
---

Welcome back to this series about digits recognition! Today, we'll directly dive in the center of our topic, by implementing the neural network.

After a quick recap about what we want to achieve, we will initialise our `Cargo` project and start coding a `struct` representing our neural network.

## The goal
### The input
As you will see in Part 2, the images we'll be working with have a `28px` side. So, to be able to recognise the digits, we will need a kind of "machine" that takes `28px` images as an input. In our case, we don't care much about color, so we'll work with grayscale images. This means that each pixel have a value somewhere between `0` and `1`, `0` being the darkest black and `1` being the lighest white. Hence, the machine we need to build takes as an input `28*28 = 784` floating numbers between `0` and `1`.

### The output


## The parts of our Neural Network
### The layers

### But how does a neural network propagate activation?

### Weights and biases

### The activation function

## Let's code!
### Project initialisation

### The `NeuralNetwork` struct

### Network initialisation
#### Random weights and biases

#### Activation function

### Feeding forward
#### Function outline
#### Unoptimised linear algebra

### Prediction


## Wrapping it up
### What we've done

### What we will do next time