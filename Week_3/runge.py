import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Runge function
def runge(x):
    return 1 / (1 + 25 * x**2)

# Derivative of Runge function
def runge_prime(x):
    return -50 * x / (1 + 25 * x**2)**2

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

# NN derivative via autodiff (keep model fixed; just differentiate w.r.t. input)
x_dense_tf = tf.convert_to_tensor(x_dense, dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x_dense_tf)
    y_dense_tf = model(x_dense_tf, training=False)        # shape (N, 1)
dy_dx_pred = tape.gradient(y_dense_tf, x_dense_tf).numpy()  # shape (N, 1)

# True derivative
dy_dx_true = runge_prime(x_dense)  # numpy shape (N, 1)

# Derivative errors (dense grid)
deriv_abs_err = np.abs(dy_dx_true - dy_dx_pred)
deriv_l2 = np.sqrt(np.mean((dy_dx_true - dy_dx_pred)**2))
deriv_linf = np.max(deriv_abs_err)

print(f"Derivative RMSE on dense grid: {deriv_l2:.6e}")
print(f"Derivative max abs error on dense grid: {deriv_linf:.6e}")

# Plot derivative comparison
plt.figure(figsize=(7,5))
plt.plot(x_dense, dy_dx_true, label="True derivative f'(x)")
plt.plot(x_dense, dy_dx_pred, label="NN derivative d/dx model(x)")
plt.legend()
plt.title("Derivative: True vs Neural Network")
plt.savefig(os.path.join(FIG_DIR, "runge_derivative_vs_nn.png"), transparent=True)

# Plot derivative absolute error
plt.figure(figsize=(7,5))
plt.plot(x_dense, deriv_abs_err)
plt.title("Absolute Error of Derivative |f'(x) - d/dx model(x)|")
plt.savefig(os.path.join(FIG_DIR, "runge_derivative_error.png"), transparent=True)

x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x_test_tf)
    y_test_tf = model(x_test_tf, training=False)
dy_dx_test_pred = tape.gradient(y_test_tf, x_test_tf).numpy()
dy_dx_test_true = runge_prime(x_test)

test_deriv_rmse = np.sqrt(np.mean((dy_dx_test_true - dy_dx_test_pred)**2))
test_deriv_linf = np.max(np.abs(dy_dx_test_true - dy_dx_test_pred))
print(f"Derivative RMSE on test set: {test_deriv_rmse:.6e}")
print(f"Derivative max abs error on test set: {test_deriv_linf:.6e}")