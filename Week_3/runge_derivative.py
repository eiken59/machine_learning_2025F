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

# ----------------------------
# Runge function and derivative
# ----------------------------
def runge(x):
    return 1 / (1 + 25 * x**2)

def runge_prime(x):
    # numpy version (for evaluation/plots)
    return -50 * x / (1 + 25 * x**2)**2

@tf.function
def runge_prime_tf(x):
    # tensorflow version (for loss)
    return -50.0 * x / tf.pow(1.0 + 25.0 * tf.pow(x, 2.0), 2.0)

# ----------------------------
# Data generation & splits
# ----------------------------
N = 100
x = np.linspace(-1, 1, N).reshape(-1, 1).astype(np.float32)
y = runge(x).astype(np.float32)

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=4)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=4)

# ----------------------------
# Base model (same structure)
# ----------------------------
net = keras.Sequential([
    layers.Dense(64, activation="tanh", input_shape=(1,)),
    layers.Dense(64, activation="tanh"),
    layers.Dense(1)
])

# ----------------------------
# Wrapper with derivative loss
# ----------------------------
class DerivWrapper(keras.Model):
    def __init__(self, base_net, lambda_deriv=1.0):
        super().__init__()
        self.net = base_net
        self.lambda_deriv = tf.constant(lambda_deriv, dtype=tf.float32)
        # trackers for history
        self.loss_tracker    = keras.metrics.Mean(name="loss")
        self.f_loss_tracker  = keras.metrics.Mean(name="f_loss")
        self.d_loss_tracker  = keras.metrics.Mean(name="d_loss")
        self.mse = keras.losses.MeanSquaredError()

    @property
    def metrics(self):
        # Keras will reset these each epoch and log them
        return [self.loss_tracker, self.f_loss_tracker, self.d_loss_tracker]

    def call(self, x, training=False):
        return self.net(x, training=training)

    def train_step(self, data):
        x_batch, y_batch = data
        x_batch = tf.cast(x_batch, tf.float32)
        y_batch = tf.cast(y_batch, tf.float32)

        with tf.GradientTape() as tape_outer:
            tape_outer.watch(x_batch)
            y_pred = self.net(x_batch, training=True)

            # function loss
            f_loss = self.mse(y_batch, y_pred)

            # derivative loss via autodiff wrt inputs
            dy_dx_pred = tape_outer.gradient(y_pred, x_batch)
            dy_dx_true = runge_prime_tf(x_batch)
            d_loss = self.mse(dy_dx_true, dy_dx_pred)

            total_loss = f_loss + self.lambda_deriv * d_loss

        grads = tf.GradientTape().gradient if False else None  # (no-op line for clarity)
        grads = tf.gradients(total_loss, self.net.trainable_variables)  # deprecated in eager
        # Use proper tape to get grads
        with tf.GradientTape() as tape:
            tape.watch(self.net.trainable_variables)
            # Recompute forward for grads (cheap for small nets). Alternatively, nest tapes.
            y_pred2 = self.net(x_batch, training=True)
            f_loss2 = self.mse(y_batch, y_pred2)
            dy_dx_pred2 = tf.gradients(y_pred2, x_batch)[0]
            d_loss2 = self.mse(runge_prime_tf(x_batch), dy_dx_pred2)
            total_loss2 = f_loss2 + self.lambda_deriv * d_loss2
        grads = tape.gradient(total_loss2, self.net.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

        # log
        self.loss_tracker.update_state(total_loss)
        self.f_loss_tracker.update_state(f_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {"loss": self.loss_tracker.result(),
                "f_loss": self.f_loss_tracker.result(),
                "d_loss": self.d_loss_tracker.result()}

    def test_step(self, data):
        x_batch, y_batch = data
        x_batch = tf.cast(x_batch, tf.float32)
        y_batch = tf.cast(y_batch, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_batch)
            y_pred = self.net(x_batch, training=False)
            f_loss = self.mse(y_batch, y_pred)
            dy_dx_pred = tape.gradient(y_pred, x_batch)
            d_loss = self.mse(runge_prime_tf(x_batch), dy_dx_pred)
            total_loss = f_loss + self.lambda_deriv * d_loss

        # Return dict; Keras will prefix these with 'val_' during fit(validation_data=...)
        return {"loss": total_loss, "f_loss": f_loss, "d_loss": d_loss}

# Build wrapper and compile
LAMBDA_DERIV = 1.0
model = DerivWrapper(net, lambda_deriv=LAMBDA_DERIV)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))

# ----------------------------
# Train (now logs f_loss, d_loss too)
# ----------------------------
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=500,
    verbose=1
)

# ----------------------------
# Dense grid predictions
# ----------------------------
x_dense = np.linspace(-1, 1, 500).reshape(-1, 1).astype(np.float32)
y_dense_pred = model(x_dense, training=False).numpy()

# Derivative via autodiff on dense grid
x_dense_tf = tf.convert_to_tensor(x_dense, dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x_dense_tf)
    y_dense_tf = model(x_dense_tf, training=False)
dy_dx_pred = tape.gradient(y_dense_tf, x_dense_tf).numpy()
dy_dx_true = runge_prime(x_dense)

# ----------------------------
# Plots: function fit
# ----------------------------
plt.figure(figsize=(7,5))
plt.plot(x_dense, runge(x_dense), label="True Runge f(x)")
plt.plot(x_dense, y_dense_pred, label="NN prediction")
plt.scatter(x_train, y_train, s=10, c="gray", alpha=0.5, label="Training data")
plt.legend()
plt.title("Runge Function Approximation (with derivative loss)")
plt.savefig(os.path.join(FIG_DIR, "runge_approx.png"), transparent=True)

# Derivative plot
plt.figure(figsize=(7,5))
plt.plot(x_dense, dy_dx_true, label="True derivative f'(x)")
plt.plot(x_dense, dy_dx_pred, label="NN derivative d/dx model(x)")
plt.legend()
plt.title("Derivative Approximation")
plt.savefig(os.path.join(FIG_DIR, "runge_derivative_vs_nn.png"), transparent=True)

# Loss curves (total, f, d)
plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="Train total")
plt.plot(history.history["f_loss"], label="Train function")
plt.plot(history.history["d_loss"], label="Train derivative")
plt.plot(history.history["val_loss"], label="Val total")
plt.plot(history.history["val_f_loss"], label="Val function")
plt.plot(history.history["val_d_loss"], label="Val derivative")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title(f"Loss Curves (λ_deriv = {LAMBDA_DERIV})")
plt.savefig(os.path.join(FIG_DIR, "loss_curves.png"), transparent=True)

# ----------------------------
# Test-set metrics (function & derivative)
# ----------------------------
# Function errors on test
y_test_pred = model(x_test, training=False).numpy()
test_f_mse = np.mean((y_test - y_test_pred)**2)
test_f_linf = np.max(np.abs(y_test - y_test_pred))

# Derivative errors on test
x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x_test_tf)
    y_test_tf = model(x_test_tf, training=False)
dy_dx_test_pred = tape.gradient(y_test_tf, x_test_tf).numpy()
dy_dx_test_true = runge_prime(x_test)
test_d_mse = np.mean((dy_dx_test_true - dy_dx_test_pred)**2)
test_d_linf = np.max(np.abs(dy_dx_test_true - dy_dx_test_pred))

print(f"Test Function   MSE: {test_f_mse:.6e},   Max|err|: {test_f_linf:.6e}")
print(f"Test Derivative MSE: {test_d_mse:.6e},   Max|err|: {test_d_linf:.6e}")