import os
import numpy as np
import matplotlib.pyplot as plt

x_train = np.load(os.path.join("./data", "digitdata", "x_train.npy"))
y_train = np.load(os.path.join("./data", "digitdata", "y_train.npy"))
x_val = np.load(os.path.join("./data", "digitdata", "x_val.npy"))
y_val = np.load(os.path.join("./data", "digitdata", "y_val.npy"))
x_test = np.load(os.path.join("./data", "digitdata", "x_test.npy"))
y_test = np.load(os.path.join("./data", "digitdata", "y_test.npy"))

# Display shapes of the data
print("Shapes of the data:")
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Visualise an image
image_index = 4
plt.imshow(x_train[image_index], cmap="gray")
plt.title(f"Label: {y_train[image_index]}")
plt.show()
