"""
Task 2: Loss Functions & Hyperparameter Tuning
CS5720 - Home Assignment 1

This script computes and compares Mean Squared Error (MSE) and
Categorical Cross-Entropy (CCE) losses, and shows how loss values
change when predictions are modified slightly.
"""

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Define true values (y_true) and model predictions (y_pred)
# One-hot encoded true labels for a 5-class classification example
y_true = tf.constant([[0, 0, 1, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0]], dtype=tf.float32)

y_pred = tf.constant([[0.05, 0.05, 0.80, 0.05, 0.05],
                       [0.70, 0.10, 0.10, 0.05, 0.05],
                       [0.10, 0.10, 0.10, 0.60, 0.10]], dtype=tf.float32)

# 2. Compute MSE and Categorical Cross-Entropy losses
mse = tf.keras.losses.MeanSquaredError()
cce = tf.keras.losses.CategoricalCrossentropy()

mse_loss = mse(y_true, y_pred).numpy()
cce_loss = cce(y_true, y_pred).numpy()

print("Original Predictions:")
print("MSE Loss:", mse_loss)
print("Categorical Cross-Entropy Loss:", cce_loss)

# 3. Modify predictions slightly and check how loss values change
# Slightly worse predictions (less confident, more spread out)
y_pred_modified = tf.constant([[0.15, 0.15, 0.40, 0.15, 0.15],
                                [0.40, 0.20, 0.20, 0.10, 0.10],
                                [0.20, 0.15, 0.15, 0.35, 0.15]], dtype=tf.float32)

mse_loss_mod = mse(y_true, y_pred_modified).numpy()
cce_loss_mod = cce(y_true, y_pred_modified).numpy()

print("\nModified (less confident) Predictions:")
print("MSE Loss:", mse_loss_mod)
print("Categorical Cross-Entropy Loss:", cce_loss_mod)

print("""
Observation:
------------
When predictions become less confident / less accurate (moving probability
mass away from the correct class), both MSE and Cross-Entropy loss
increase. Cross-Entropy tends to change more sharply than MSE for
classification-style predictions because it heavily penalizes confident
wrong predictions and is specifically designed to measure the difference
between probability distributions, whereas MSE treats every output
independently and changes more gradually.
""")

# 4. Plot loss function values using Matplotlib
labels = ["Original Predictions", "Modified Predictions"]
mse_values = [mse_loss, mse_loss_mod]
cce_values = [cce_loss, cce_loss_mod]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
bars1 = ax.bar([i - width/2 for i in x], mse_values, width, label="MSE Loss", color="#4C72B0")
bars2 = ax.bar([i + width/2 for i in x], cce_values, width, label="Categorical Cross-Entropy Loss", color="#DD8452")

ax.set_ylabel("Loss Value")
ax.set_title("Comparison of MSE vs Categorical Cross-Entropy Loss")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=150)
print("\nBar chart saved as loss_comparison.png")
