import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Runge function
def runge(x):
    return 1 / (1 + 25 * x**2)

# Data generation
N = 100
x = np.linspace(-1, 1, N).reshape(-1, 1)
y = runge(x)

# Split into train (70%) and temp (30%)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=4)
# Split temp into validation (15%) and test (15%)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=4)

# Build model
model = keras.Sequential([
    layers.Dense(64, activation="tanh", input_shape=(1,)),
    layers.Dense(64, activation="tanh"),
    layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss="mse")

# Train
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=500,
    verbose=1
)

x_dense = np.linspace(-1, 1, 500).reshape(-1, 1)
y_dense_pred = model.predict(x_dense)

# Plot Runge function vs NN approximation
plt.figure(figsize=(7,5))
plt.plot(x_dense, runge(x_dense), label="True Runge function")
plt.plot(x_dense, y_dense_pred, label="Neural Network Approximation")
plt.scatter(x_train, y_train, s=10, c="gray", alpha=0.5, label="Training data")
plt.legend()
plt.title("Runge Function Approximation")
plt.savefig(os.path.join(FIG_DIR, "runge_approx.png"), transparent=True)
# plt.show()

# Plot the loss curves
plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE (log scale)")
plt.legend()
plt.title("Loss Curves")
plt.savefig(os.path.join(FIG_DIR, "loss_curves.png"), transparent=True)
# plt.show()

# Test set
test_mse = model.evaluate(x_test, y_test, verbose=0)
y_test_pred = model.predict(x_test)
max_err = np.max(np.abs(y_test - y_test_pred))

print(f"Test MSE: {test_mse:.6f}")
print(f"Max error on test set: {max_err:.6f}")