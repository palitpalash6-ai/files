"""
Task 3: Train a Model with Different Optimizers
CS5720 - Home Assignment 1

This script trains two identical neural network models on the MNIST
dataset -- one using the Adam optimizer and one using SGD -- then
compares their training and validation accuracy trends.
"""

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to the 0-1 range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


def build_model():
    """Builds a simple feedforward neural network for MNIST classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    return model


EPOCHS = 5

# 2. Train two models: one with Adam and another with SGD
print("Training model with Adam optimizer...")
model_adam = build_model()
model_adam.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
history_adam = model_adam.fit(x_train, y_train,
                               validation_data=(x_test, y_test),
                               epochs=EPOCHS, verbose=2)

print("\nTraining model with SGD optimizer...")
model_sgd = build_model()
model_sgd.compile(optimizer="sgd",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
history_sgd = model_sgd.fit(x_train, y_train,
                             validation_data=(x_test, y_test),
                             epochs=EPOCHS, verbose=2)

# 3. Compare training and validation accuracy trends
print("\nFinal Adam - Train Acc: {:.4f}, Val Acc: {:.4f}".format(
    history_adam.history["accuracy"][-1], history_adam.history["val_accuracy"][-1]))
print("Final SGD  - Train Acc: {:.4f}, Val Acc: {:.4f}".format(
    history_sgd.history["accuracy"][-1], history_sgd.history["val_accuracy"][-1]))

# Plot accuracy comparison
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, history_adam.history["accuracy"], label="Adam - Train Acc", marker="o")
plt.plot(epochs_range, history_adam.history["val_accuracy"], label="Adam - Val Acc", marker="o", linestyle="--")
plt.plot(epochs_range, history_sgd.history["accuracy"], label="SGD - Train Acc", marker="s")
plt.plot(epochs_range, history_sgd.history["val_accuracy"], label="SGD - Val Acc", marker="s", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Adam vs SGD: Training and Validation Accuracy on MNIST")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("adam_vs_sgd_accuracy.png", dpi=150)
print("\nAccuracy comparison chart saved as adam_vs_sgd_accuracy.png")
