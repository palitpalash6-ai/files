"""
Task 1: Tensor Manipulations & Reshaping
CS5720 - Home Assignment 1

This script demonstrates basic tensor creation, inspection, reshaping,
transposing, and broadcasting using TensorFlow.
"""

import tensorflow as tf

# 1. Create a random tensor of shape (4, 6)
tensor = tf.random.uniform(shape=(4, 6), minval=0, maxval=10)
print("Original Tensor:")
print(tensor.numpy())

# 2. Find its rank and shape using TensorFlow functions
original_rank = tf.rank(tensor)
original_shape = tf.shape(tensor)
print("\nOriginal Rank:", original_rank.numpy())
print("Original Shape:", original_shape.numpy())

# 3. Reshape it into (2, 3, 4) and transpose it to (3, 2, 4)
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))
print("\nReshaped Tensor shape (2, 3, 4):", reshaped_tensor.shape)
print("Reshaped Rank:", tf.rank(reshaped_tensor).numpy())

transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
print("\nTransposed Tensor shape (3, 2, 4):", transposed_tensor.shape)
print("Transposed Rank:", tf.rank(transposed_tensor).numpy())

# 4. Broadcast a smaller tensor (1, 4) to match the larger tensor and add them
small_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0]])  # shape (1, 4)
print("\nSmall Tensor shape:", small_tensor.shape)

# Broadcasting automatically expands (1, 4) to match (3, 2, 4) during addition
broadcast_result = transposed_tensor + small_tensor
print("Broadcast Result shape:", broadcast_result.shape)
print("Broadcast Result:")
print(broadcast_result.numpy())

# 5. Explanation of broadcasting
print("""
Broadcasting Explanation:
--------------------------
Broadcasting lets TensorFlow perform element-wise operations on tensors
of different shapes without explicitly copying data. When operating on
two tensors, TensorFlow compares their shapes starting from the
rightmost (last) dimension:

  1. If two dimensions are equal, they are compatible as-is.
  2. If one of the dimensions is 1, that dimension is "stretched"
     (virtually repeated) to match the other tensor's dimension.
  3. If a tensor has fewer dimensions, TensorFlow pads its shape with
     1s on the left until both tensors have the same number of
     dimensions, then applies rule 1 and 2.

In this example, the small tensor of shape (1, 4) is broadcast against
a tensor of shape (3, 2, 4). TensorFlow pads (1, 4) to (1, 1, 4), then
stretches the size-1 dimensions to match (3, 2, 4), effectively
repeating the small tensor's row across every position in the larger
tensor before adding them element-wise. No extra memory copies are
made explicitly by the programmer -- TensorFlow handles the expansion
internally and efficiently.
""")
