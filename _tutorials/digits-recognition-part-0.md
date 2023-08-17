---
title: "Handwritten digits recognition from scratch in Rust - Part 0: Introduction"
excerpt: "The introduction of the series: we'll discuss the differents steps to recognise digits."
permalink: /digits-recognition-part-0/
toc: true
toc_label: "Introduction"
tags:
    - deep learning
---

Welcome! In this series, we will implement a detector capable of recognising handwritten digits.

![Digits examples](../assets/projects/digits-mlp.png)

The task is simple: given a grayscale image of `28px` side representing a handwritten digit, we want to return a digit, from `0` to `9`, that corresponds to the digit on the image. To do so, we will use a **neural network**, and train it on a huge dataset.


## Prerequisites
No advanced knowledge on neural networks is necessary to follow this series. I still recommend basic understanding of what a neural network is. If you don't know what a neural network is, or if you need a refresh, I suggest the incredible [3blue1brown's video series on neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). Seriously, it's probably the best explanation on neural networks one could find.

{% include video id="aircAruvnKk" provider="youtube" %}

Furthermore, the code for this tutorial will be written in Rust. Once again, nothing too advanced, but a basic understanding of Rust is recommended. I will explain most details as we move forward, but understanding the code you'll write is the best way to learn. 

We will only use Rust's basic features such as `struct`, `enum`, and `Vector`. If you're familiar with these concepts, you'll be able to follow along.
{: .notice--info}

## Steps of this tutorial
### Part 1: The Neural Network
I will detail the neural network we will use, and we will implement from scratch a neural network, randomly initialised. We will also code the central function called `feed_forward`, that almost magically transforms our input into an output.

### Part 2: Network evaluation
We will use the MNIST dataset to load some actual images, using the `mnist` crate. We will write some functions to test the network we just build... and see how *poorly* it performs just yet. That will obviously lead us to...

### Part 3: Network training
We will see how to transform a random dumb network into a hopefully-smart digit recogniser. We will also train the network on a few images to check that everything is working before the big finale.

### Part 4: Saving and real-size training
This is where we'll discover how long the training process actually takes, and why we need to be able to load and save trained networks. Finally, we will have a working network, hopefully with some pretty sweet accuracy. *We'll run some benchmarks to flex.*

That being said, we are ready to begin our journey in the world of deep learning. See you next time!