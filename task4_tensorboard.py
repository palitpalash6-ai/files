"""
Task 4: Train a Neural Network and Log to TensorBoard
CS5720 - Home Assignment 1

This script loads MNIST, trains a simple feedforward neural network for
5 epochs, and logs training/validation loss and accuracy to TensorBoard.
"""

import datetime
import tensorflow as tf

# 1. Load the MNIST dataset and preprocess it
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 2. Train a simple neural network model and enable TensorBoard logging
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Create a unique log directory for this run, timestamped, inside logs/fit/
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=5,
          callbacks=[tensorboard_callback])

print(f"\nTraining complete. Logs saved to: {log_dir}")
print("To view TensorBoard, run this command from the same directory:")
print("    tensorboard --logdir logs/fit")
print("Then open the printed local URL (usually http://localhost:6006) in your browser.")
