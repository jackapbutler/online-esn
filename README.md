# online-esn
Echo State Networks (ESN) provide an architecture and supervised learning principle for more energy efficient recurrent neural networks (RNNs). This repository implements an ESN along with a variety of different online learning algorithms for temporal classification tasks.

The main idea is:

1. Drive a random, sparsely connected, fixed recurrent neural network with the input signal, thereby inducing in each neuron within this reservoir network a non-linear response signal.
    
2. Combine a desired output signal (labels) by a trainable parametric combination of all of these response signals.

You can see an example workflow for the Ti46 dataset in [`example.ipynb`](example.ipnb).

# System
The below figure outlines an example of classifying columnwise Mnist digits using a single linear output layer. This implementation can achieve upwards of 95% on columnwise Mnist when the ESN is combined with a two layer MLP.

<span style="display:block;text-align:center">![Image of framework](transformation-example.png)</span>

# Formatting
This repository uses `black`, `mypy` and `isort` for formatting the codebase.
