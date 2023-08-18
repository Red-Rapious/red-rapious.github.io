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

## The parts of our Neural Network
### The input
As you will see in Part 2, the images we'll be working with have a `28px` side. So, to be able to recognise the digits, we will need a kind of "machine" that takes `28px` images as an input. In our case, we don't care much about color, so we'll work with grayscale images. This means that each pixel have a value somewhere between `0` and `1`, `0` being the darkest black and `1` being the lighest white. Hence, the machine we need to build takes as an input `28*28 = 784` floating numbers between `0` and `1`.

![Image as input](../assets/tutorials/digits-recognition/pixels-as-input-inverted.png)
Our image as an input. *(Credit: 3blue1brown)*
{: .text-center}

Code-wise, our neural network will then take as an input a `Vec<f64>`, of size `784`.

**Note:** For this tutorial, I decided to use `Vector`s to represent many structures. There's multiple alternatives, like using arrays, or even vectors and matrices from the [`nalgebra`](https://www.nalgebra.org) crate (more on that later). For the sake of simplicity, we'll stick with `Vector`s since they can be dynamically allocated.
{: .notice--info}

### The output
To be fair, there's multiple ways to represent the output of our machine. We want it to return an integer between `0` and `9`, but neural networks, as we'll see, don't work much with integers, but instead with floating numbers.

So how can our neural network tell us which digit it detected? By using `10` outputs, one corresponding to each digit. We will then check which output is the most "activated", and deduce the corresponding digit. In fancy terms, we will take the [`argmax`](https://en.wikipedia.org/wiki/Arg_max) of the output layer.

### The intermediate layers
The choice of the number and sizes of intermediate layers is kind of... random? As long as the intermediate layers are not too small nor too large, we should be good. And one great thing about coding, is that you can adjust things up at the end. 

Hence, I chose **two** intermediate layers, of **`16` neurons each** (kind of a reference to 3blue1brown's video), but out implementation of the neural network will take these as an argument, making it *super-convenient* to change later on. 

To do so, we'll store the size of each layer inside a `Vec<usize>`. If the intermediate layers can be changed by the "user", we'll have to make sure that the first element of the `Vec` is a layer of `784` input neurons, and that the last one is a layer of `10` output neurons.

### But how does a neural network propagate activation?
> TODO

### Weights and biases
That being said, we will represent the weights as matrices, and biases as vectors (in the linear algebra way). To keep things consistent, we will have a `Vec`-only approach of representing our weights and biases. This means that the weights between two layers will be a `Vec<Vec<f64>>`, and that the biases will be a `Vec<f64>`.

**Note:** Hand-coding the linear algebra part is, in my opinion, the best way to keep it simple. Nevertheless, if you're feeling confident and not afraid to adapt some code I'll write, using [`nalgebra`](https://www.nalgebra.org/docs/user_guide/getting_started), a linear algebra library, will be much more efficient, and you won't have to write some boring matrix multiplication functions.
{: .notice--info}

### The activation function
To make things a bit more general, our neural network object will take the activation function as an argument. We will only implement two different ones, [`Sigmoid`](https://en.wikipedia.org/wiki/Sigmoid_function) and [`ReLU`](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)), but we'll make sure that adding a new one is quite simple.

![Sigmoid and ReLU](../assets/tutorials/digits-recognition/sigmoid-relu.png)
The graphs of Sigmoid and ReLU.
{: .text-center}

## Let's code!
Enough tchit-tchat, let's get to the fun part, shall we?
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