"""
Temperature grid processing for CWA O-A0038-003

This script does the following:
1) Parse the provided XML file `O-A0038-003.xml` to extract the 67x120 temperature grid
	and geographic metadata (bottom-left, top-right, resolution).
2) Build two supervised datasets:
	- Classification: (longitude, latitude, label), where label=1 if value != -999 else 0
	- Regression: (longitude, latitude, value), keeping only valid values (value != -999)
3) Train models with a 70/15/15 train/val/test split:
	- Classification: default is Decision Tree; logistic regression is also available.
	- Regression: default is a TensorFlow neural network; KNN and sklearn MLP are available options.
4) Generate visualizations and save CSVs + figures under Week_4.

Notes:
- Grid is 67 (lon) x 120 (lat), resolution 0.03 degrees.
- Longitude increases west->east along each row, then latitude increases south->north by rows.
"""

import argparse
import os
import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	mean_absolute_error,
	mean_squared_error,
	r2_score,
	precision_recall_curve,
	precision_score,
	recall_score,
	f1_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import tensorflow as tf


# ---------------------------
# Parsing helpers
# ---------------------------
def parse_xml(xml_path: str) -> Tuple[dict, np.ndarray]:
	"""
	Parse the CWA XML and return metadata and the temperature grid as a numpy array
	with shape (ny, nx) where ny=120 (lat), nx=67 (lon).
	"""
	tree = ET.parse(xml_path)
	root = tree.getroot()
	ns = {"ns": "urn:cwa:gov:tw:cwacommon:0.1"}

	# Extract geographic metadata
	meta = {
		"BottomLeftLongitude": float(root.findtext("ns:dataset/ns:GeoInfo/ns:BottomLeftLongitude", namespaces=ns)),
		"BottomLeftLatitude": float(root.findtext("ns:dataset/ns:GeoInfo/ns:BottomLeftLatitude", namespaces=ns)),
		"TopRightLongitude": float(root.findtext("ns:dataset/ns:GeoInfo/ns:TopRightLongitude", namespaces=ns)),
		"TopRightLatitude": float(root.findtext("ns:dataset/ns:GeoInfo/ns:TopRightLatitude", namespaces=ns)),
		"nx": 67,
		"ny": 120,
		"resolution": 0.03,
	}

	# Read the flat content string of temps
	content_node = root.find("ns:dataset/ns:Resource/ns:Content", namespaces=ns)
	if content_node is None or not content_node.text:
		raise ValueError("Content node is missing in the XML file.")

	# Some lines are wrapped; unify by replacing newlines with commas and splitting
	raw = content_node.text.strip().replace("\n", ",")
	tokens = [t.strip() for t in raw.split(",") if t.strip() != ""]

	values = []
	for t in tokens:
		try:
			values.append(float(t))  # handles forms like 28.1E+00 or -999.0E+00
		except ValueError:
			# Skip malformed tokens gracefully
			continue

	expected = meta["nx"] * meta["ny"]
	if len(values) != expected:
		raise ValueError(f"Parsed values count {len(values)} != expected {expected}")

	grid = np.array(values, dtype=float).reshape(meta["ny"], meta["nx"])  # (lat, lon)
	return meta, grid


def build_lon_lat(meta: dict) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Build 1D arrays of longitudes and latitudes given metadata.
	"""
	lons = meta["BottomLeftLongitude"] + np.arange(meta["nx"]) * meta["resolution"]
	lats = meta["BottomLeftLatitude"] + np.arange(meta["ny"]) * meta["resolution"]
	return lons, lats


# ---------------------------
# Dataset builders
# ---------------------------
def make_datasets(meta: dict, grid: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Create classification and regression datasets from the grid.

	Classification: (longitude, latitude, label) where label=1 if value != -999 else 0
	Regression:     (longitude, latitude, value) only where value != -999
	"""
	lons, lats = build_lon_lat(meta)
	lon_grid, lat_grid = np.meshgrid(lons, lats)  # both shape (ny, nx)

	values = grid.flatten()
	lon_flat = lon_grid.flatten()
	lat_flat = lat_grid.flatten()

	# Label is 1 if valid; invalid is exactly -999.0 per spec
	labels = (values != -999.0).astype(int)
	cls_df = pd.DataFrame({
		"longitude": lon_flat,
		"latitude": lat_flat,
		"label": labels,
	})

	# Keep only valid rows for regression
	valid_mask = values != -999.0
	reg_df = pd.DataFrame({
		"longitude": lon_flat[valid_mask],
		"latitude": lat_flat[valid_mask],
		"value": values[valid_mask],
	})

	return cls_df, reg_df


# ---------------------------
# Visualizations
# ---------------------------
def plot_validity_and_temp(meta: dict, grid: np.ndarray, fig_dir: str) -> Tuple[str, str]:
	"""
	Save validity map and temperature heatmap figures.
	Returns the paths to the saved figures.
	"""
	os.makedirs(fig_dir, exist_ok=True)
	lons, lats = build_lon_lat(meta)
	mask_valid = (grid != -999.0)

	# Validity map (1 valid, 0 invalid)
	plt.figure(figsize=(5, 8))
	# Discrete colormap: 0 -> white, 1 -> black
	discrete_cmap = ListedColormap(["white", "black"])
	norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=discrete_cmap.N)
	plt.imshow(
		mask_valid.astype(int),
		origin="lower",
		extent=[lons.min(), lons.max(), lats.min(), lats.max()],
		aspect="auto",
		cmap=discrete_cmap,
		norm=norm,
		interpolation="nearest",
	)
	# Legend with square patches instead of continuous colorbar
	legend_handles = [
		Patch(facecolor="white", edgecolor="black", label="Invalid (0)"),
		Patch(facecolor="black", edgecolor="black", label="Valid (1)"),
	]
	plt.legend(handles=legend_handles, title="Validity", loc="lower right", frameon=True)
	plt.xlabel("Longitude")
	plt.ylabel("Latitude")
	plt.title("Data Validity Map")
	valid_path = os.path.join(fig_dir, "validity_map.png")
	plt.tight_layout(); plt.savefig(valid_path, dpi=150); plt.close()

	# Temperature heatmap (NaN where invalid)
	temp = np.where(mask_valid, grid, np.nan)
	plt.figure(figsize=(6, 8))
	im = plt.imshow(temp, origin="lower",
					extent=[lons.min(), lons.max(), lats.min(), lats.max()],
					aspect="auto", cmap="turbo")
	plt.colorbar(im, label="Temperature (°C)")
	plt.xlabel("Longitude")
	plt.ylabel("Latitude")
	plt.title("Temperature Heatmap (valid only)")
	temp_path = os.path.join(fig_dir, "temperature_heatmap.png")
	plt.tight_layout(); plt.savefig(temp_path, dpi=150); plt.close()

	return valid_path, temp_path


def plot_confusion_matrix(cm: np.ndarray, fig_dir: str) -> str:
	"""Save a confusion matrix heatmap and return the file path."""
	os.makedirs(fig_dir, exist_ok=True)
	plt.figure(figsize=(4, 4))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.title("Confusion Matrix")
	path = os.path.join(fig_dir, "confusion_matrix.png")
	plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
	return path


def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, fig_dir: str, suffix: str = "") -> str:
	"""Save a scatter plot of y_true vs y_pred and return the file path."""
	os.makedirs(fig_dir, exist_ok=True)
	plt.figure(figsize=(5, 5))
	plt.scatter(y_true, y_pred, s=8, alpha=0.6)
	vmin = float(min(y_true.min(), y_pred.min()))
	vmax = float(max(y_true.max(), y_pred.max()))
	plt.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1)
	plt.xlabel("True Temperature (°C)")
	plt.ylabel("Predicted Temperature (°C)")
	plt.title("Regression: True vs Predicted")
	name = "regression_true_vs_pred" + (f"_{suffix}" if suffix else "") + ".png"
	path = os.path.join(fig_dir, name)
	plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
	return path


# ---------------------------
# Models
# ---------------------------
def train_classification(cls_df: pd.DataFrame, model: str = "dt", fig_dir: str | None = None, threshold: float = 0.5) -> Tuple[dict, np.ndarray]:
	"""
	Train a classification model on (longitude, latitude) -> label with 70/15/15 split.
	model in {"logreg", "dt"}. Returns metrics on TEST split and its confusion matrix.
	"""
	X = cls_df[["longitude", "latitude"]].values
	y = cls_df["label"].values

	# 70/15/15 split (train/val/test). Use stratify to respect class imbalance.
	X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.15, random_state=4, stratify=y)
	X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.1764706, random_state=4, stratify=y_tmp)
	# Note: 0.1764706 of the remaining 85% ≈ 15% absolute, giving ~70/15/15.

	if model == "logreg":
		# Logistic Regression with class imbalance handling
		clf = LogisticRegression(max_iter=500, class_weight="balanced")
		clf.fit(X_tr, y_tr)
		# Tune threshold on validation set to maximize F1
		y_prob_val = clf.predict_proba(X_val)[:, 1]
		prec, rec, th = precision_recall_curve(y_val, y_prob_val)
		f1_vals = (2 * prec * rec) / (prec + rec + 1e-12)
		best_idx = int(np.nanargmax(f1_vals[:-1])) if th.size > 0 else 0
		best_thresh = float(th[best_idx]) if th.size > 0 else threshold
		# Apply on test
		y_prob = clf.predict_proba(X_te)[:, 1]
		y_pred = (y_prob >= best_thresh).astype(int)

	elif model == "dt":
		# Decision Tree baseline (no projection) with class imbalance handling
		clf = DecisionTreeClassifier(random_state=4, max_depth=8, class_weight="balanced")
		clf.fit(X_tr, y_tr)
		y_pred = clf.predict(X_te)

	else:
		raise ValueError("Unknown clf model: " + str(model))

	acc = accuracy_score(y_te, y_pred)
	prec_macro = precision_score(y_te, y_pred, zero_division=0)
	rec_macro = recall_score(y_te, y_pred, zero_division=0)
	f1 = f1_score(y_te, y_pred, zero_division=0)
	cm = confusion_matrix(y_te, y_pred)
	metrics = {
		"accuracy": acc,
		"precision": prec_macro,
		"recall": rec_macro,
		"f1": f1,
		"n_train": int(len(y_tr)),
		"n_val": int(len(y_val)),
		"n_test": int(len(y_te)),
	}
	return metrics, cm


def train_regression(reg_df: pd.DataFrame, model: str = "tf") -> Tuple[dict, np.ndarray, np.ndarray]:
	"""
	Train a regression model on (longitude, latitude) -> value with 70/15/15 split.
	model in {"tf", "knn", "mlp"}. Returns metrics on TEST split and (y_true, y_pred) for plotting.
	"""
	X = reg_df[["longitude", "latitude"]].values
	y = reg_df["value"].values

	# 70/15/15 split (no stratify for regression)
	X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.15, random_state=4)
	X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.1764706, random_state=4)

	if model == "tf":
		# TensorFlow MLP: scale inputs, train with val set + early stopping
		scaler = StandardScaler()
		X_tr_s = scaler.fit_transform(X_tr)
		X_val_s = scaler.transform(X_val)
		X_te_s = scaler.transform(X_te)

		model_tf = tf.keras.Sequential([
			tf.keras.layers.Input(shape=(2,)),
			tf.keras.layers.Dense(64, activation="sigmoid"),
			tf.keras.layers.Dense(64, activation="sigmoid"),
			tf.keras.layers.Dense(1)
		])
		model_tf.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss="mse", metrics=["mae"])
		es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
		model_tf.fit(X_tr_s, y_tr, validation_data=(X_val_s, y_val), epochs=5000, batch_size=64, verbose=1, callbacks=[es])
		y_pred = model_tf.predict(X_te_s, verbose=1).reshape(-1)

	elif model == "gmm":
		# Gaussian Mixture regression via joint p([x,y]) and conditional E[y|x]
		# Fit GMM on standardized joint [lon, lat, value]
		scaler_all = StandardScaler()
		Z_tr = np.column_stack([X_tr, y_tr])
		Z_tr_s = scaler_all.fit_transform(Z_tr)
		gmm = GaussianMixture(n_components=6, covariance_type="full", random_state=4, max_iter=500)
		gmm.fit(Z_tr_s)

		# Prepare standardized X for val/test using scaler_all params for first two dims
		mu_all = scaler_all.mean_
		sig_all = scaler_all.scale_
		X_te_s = (X_te - mu_all[:2]) / sig_all[:2]

		means = gmm.means_
		covs = gmm.covariances_
		weights = gmm.weights_

		mu_x = means[:, :2]
		mu_y = means[:, 2]
		# Precompute per-component inverses and A_k = S_yx S_xx^{-1}
		inv_Sxx = []
		log_dets = []
		A_list = []
		for k in range(means.shape[0]):
			Sxx = covs[k][:2, :2]
			Syx = covs[k][2, :2]
			inv = np.linalg.inv(Sxx)
			inv_Sxx.append(inv)
			log_dets.append(np.log(np.linalg.det(Sxx) + 1e-12))
			A_list.append(Syx @ inv)
		inv_Sxx = np.stack(inv_Sxx, axis=0)
		A = np.stack(A_list, axis=0)  # shape (K, 2)
		log_dets = np.asarray(log_dets)

		# Compute responsibilities gamma_k(x) ∝ w_k N(x|mu_xk, Sxx_k)
		# Evaluate log density for stability
		d = 2
		const = -0.5 * d * np.log(2 * np.pi)
		# For each sample, component
		diff = X_te_s[:, None, :] - mu_x[None, :, :]  # (n, K, 2)
		exp_term = -0.5 * np.einsum("nki,kij,nkj->nk", diff, inv_Sxx, diff)
		log_prob = np.log(weights + 1e-12)[None, :] + const - 0.5 * log_dets[None, :] + exp_term
		log_prob -= log_prob.max(axis=1, keepdims=True)
		resp = np.exp(log_prob)
		resp /= resp.sum(axis=1, keepdims=True) + 1e-12

		# Conditional mean per component: mu_y + A_k @ (x - mu_xk)
		cond_means = mu_y[None, :] + np.einsum("ki,nki->nk", A, diff)
		y_te_s = (resp * cond_means).sum(axis=1)
		# Invert scaling for y
		y_pred = y_te_s * sig_all[2] + mu_all[2]

	elif model == "mlp":
		# A small neural network; scale inputs for better conditioning
		scaler = StandardScaler()
		X_tr_s = scaler.fit_transform(X_tr)
		X_val_s = scaler.transform(X_val)
		X_te_s = scaler.transform(X_te)

		regr = MLPRegressor(hidden_layer_sizes=(64, 64), activation="sigmoid", random_state=4, max_iter=500)
		regr.fit(X_tr_s, y_tr)
		_ = regr.predict(X_val_s)
		y_pred = regr.predict(X_te_s)

	else:
		raise ValueError("Unknown reg model: " + str(model))

	mae = mean_absolute_error(y_te, y_pred)
	rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
	r2 = r2_score(y_te, y_pred)
	metrics = {"mae": mae, "rmse": rmse, "r2": r2}
	return metrics, y_te, y_pred


# ---------------------------
# Main / CLI
# ---------------------------
def main():
	parser = argparse.ArgumentParser(description="Process temperature XML, build datasets, train models, and plot figures.")
	# Respect user's BASE_DIR/FIG_DIR pattern
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	FIG_DIR = os.path.join(BASE_DIR, "figures")
	os.makedirs(FIG_DIR, exist_ok=True)

	parser.add_argument("--xml", default=os.path.join(BASE_DIR, "O-A0038-003.xml"), help="Path to O-A0038-003 XML file")
	parser.add_argument("--outdir", default=BASE_DIR, help="Output directory (default: Week_4)")
	args = parser.parse_args()

	os.makedirs(args.outdir, exist_ok=True)
	fig_dir = FIG_DIR

	# 1) Parse XML and reconstruct grid
	meta, grid = parse_xml(args.xml)

	# 2) Datasets
	cls_df, reg_df = make_datasets(meta, grid)
	cls_csv = os.path.join(args.outdir, "classification_dataset.csv")
	reg_csv = os.path.join(args.outdir, "regression_dataset.csv")
	if not os.path.exists(cls_csv): cls_df.to_csv(cls_csv, index=False)
	if not os.path.exists(reg_csv): reg_df.to_csv(reg_csv, index=False)

	# 3) Visualizations (maps)
	valid_map_path, temp_map_path = plot_validity_and_temp(meta, grid, fig_dir)

	# 4) Models + evaluation plots
	# Show class balance for visibility
	cls_counts = cls_df['label'].value_counts().to_dict()
	print("Class distribution (label=0/1):", cls_counts)

	# Always run: (1) Decision Tree classification, (2) GMM regression, (3) NN regression
	cls_metrics, cm = train_classification(cls_df, model="dt", fig_dir=fig_dir)
	reg_metrics_gmm, y_true_gmm, y_pred_gmm = train_regression(reg_df, model="gmm")
	reg_metrics_tf, y_true_tf, y_pred_tf = train_regression(reg_df, model="tf")

	cm_path = plot_confusion_matrix(cm, fig_dir)
	reg_scatter_path_gmm = plot_regression_scatter(y_true_gmm, y_pred_gmm, fig_dir, suffix="GMM")
	reg_scatter_path_tf = plot_regression_scatter(y_true_tf, y_pred_tf, fig_dir, suffix="NN")

	# Print a concise summary to console
	print("=== Summary ===")
	print(f"Splits (train/val/test): {cls_metrics['n_train']}/{cls_metrics['n_val']}/{cls_metrics['n_test']}")
	print(f"Classification (test) -> Acc: {cls_metrics['accuracy']:.4f}, F1: {cls_metrics['f1']:.4f}, Precision: {cls_metrics['precision']:.4f}, Recall: {cls_metrics['recall']:.4f}")
	print(f"Regression (GMM) -> MAE: {reg_metrics_gmm['mae']:.3f} °C, RMSE: {reg_metrics_gmm['rmse']:.3f} °C, R2: {reg_metrics_gmm['r2']:.4f}")
	print(f"Regression (NN)  -> MAE: {reg_metrics_tf['mae']:.3f} °C, RMSE: {reg_metrics_tf['rmse']:.3f} °C, R2: {reg_metrics_tf['r2']:.4f}")
	print("Saved:")
	print("-", cls_csv)
	print("-", reg_csv)
	for p in [valid_map_path, temp_map_path, cm_path, reg_scatter_path_gmm, reg_scatter_path_tf]:
		print("-", p)


if __name__ == "__main__":
	main()

