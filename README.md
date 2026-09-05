# Home Assignment 1 — Neural Networks and Deep Learning (CS5720)

**Student Name:** Palash Palit
**University of Central Missouri — Department of Computer Science & Cybersecurity**
**Course:** CS5720 Neural Network and Deep Learning — Fall 2026

## Overview

This repository contains the source code for Home Assignment 1, covering four
programming tasks built with TensorFlow/Keras:

1. **Tensor Manipulations & Reshaping** — creating, inspecting, reshaping,
   transposing, and broadcasting tensors.
2. **Loss Functions & Hyperparameter Tuning** — implementing and comparing
   Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE) loss.
3. **Train a Model with Different Optimizers** — training identical models
   on MNIST with Adam vs. SGD and comparing accuracy trends.
4. **Train a Neural Network and Log to TensorBoard** — training on MNIST
   with TensorBoard logging enabled to visualize loss/accuracy curves.

Written short answers for Part I (conceptual questions on AI/ML/DL,
neural network layers, perceptrons, and activation functions) are included
in the submitted assignment document.

## Files

| File | Description |
|---|---|
| `task1_tensor_ops.py` | Tensor creation, reshaping, transposing, and broadcasting demo |
| `task2_loss_functions.py` | MSE vs. Categorical Cross-Entropy comparison + bar chart |
| `task3_optimizers.py` | Trains MNIST models with Adam and SGD, plots accuracy comparison |
| `task4_tensorboard.py` | Trains an MNIST model with TensorBoard logging enabled |
| `loss_comparison.png` | Output chart from Task 2 |

## How to Run

Requires Python 3.9+ and the following packages:

```bash
pip install tensorflow matplotlib
```

Run each task independently:

```bash
python task1_tensor_ops.py
python task2_loss_functions.py
python task3_optimizers.py
python task4_tensorboard.py
```

Tasks 3 and 4 will automatically download the MNIST dataset on first run
(requires internet access) and may take a few minutes to train.

### Viewing TensorBoard (Task 4)

After running `task4_tensorboard.py`, launch TensorBoard from the same
directory:

```bash
tensorboard --logdir logs/fit
```

Then open the printed local URL (typically `http://localhost:6006`) in a
browser to view training/validation loss and accuracy curves.

## Notes

- All code is commented to explain each step.
- Task 3 and Task 4 use a simple feedforward network (Flatten → Dense(128,
  ReLU) → Dropout(0.2) → Dense(10, Softmax)) trained for 5 epochs.
- See the submitted assignment document for full explanations, program
  output, and answers to the TensorBoard reflection questions (4.1).

A 2–3 minute video demonstrating this assignment and explaining key code
snippets is linked in the Bright Space submission.
