---
title: "Genetic Birds Simulator"
excerpt: "A simulation of birds using Neural Networks and a Genetic Algorithm."
tags:
    - deep learning
    - genetic algorithm
    - rust
header:
  teaser: /assets/teaser-images/birds-simulator.png
---

![Genetic Birds](../assets/projects/genetic-birds.png)

This project is a kind of sequel to [MLP-Digits-Recognition](), an implementation of a neural network from scratch. This time, I used [this great tutorial](https://pwy.io/posts/learning-to-fly-pt1/) as a starting point, to train the neural network with a genetic algorithm, instead of backpropagation.

The goal was to teach birds (triangles) how to catch food (circles) by using learning mechanisms that simulate natural evolution. 

To do so, birds are represented by a small FFNN (Feed Forward Neural Network), initialized with random weights and biases. The input layer is connected to the "eye" of the bird, an array of "cells" that are activated depending on the position of nearby food. The learning process, a genetic algorithm, combines three steps: **selection** (two parents are selected), **crossover** (the genomes of parents are combined into a new one), and **mutation** (random changes to the child's genome). In this case, a bird's genome is an array of the weights and biases of its neural network.

This simulation, coded in Rust, is meant to be used using only a web browser, thanks to WebAssembly. See the instructions in the [project's README](https://github.com/Red-Rapious/Genetic-Birds-Simulator) to try it yourself.