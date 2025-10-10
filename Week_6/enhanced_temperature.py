"""
Enhanced temperature processing for Assignment 2 (piecewise regression)

We combine:
- C(x): a classification model on (lon, lat) -> {0,1}
- R(x): a regression model on (lon, lat) -> value

And define the piecewise function h(x):
  h(x) = R(x) if C(x)=1; else -999.

Implementation choices:
- C(x): DecisionTreeClassifier with 70/15/15 split and threshold tuned on validation
- R(x): Neural network regression (scikit-learn MLP), trained on valid points only
- Decision boundary plot: shows ONLY the tuned best-p boundary (clean styling), with TN/FP/FN/TP colors

Outputs (saved under Week_6/figures):
- decision_boundary_piecewise.png
- confusion_matrix_piecewise.png
- piecewise_temperature.png (heatmap of h(x) over the grid)
"""

import os
import xml.etree.ElementTree as ET
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


# ---------------------------
# Parsing helpers
# ---------------------------
def parse_xml(xml_path: str) -> Tuple[dict, np.ndarray]:
	tree = ET.parse(xml_path)
	root = tree.getroot()
	ns = {"ns": "urn:cwa:gov:tw:cwacommon:0.1"}

	meta = {
		"BottomLeftLongitude": float(root.findtext("ns:dataset/ns:GeoInfo/ns:BottomLeftLongitude", namespaces=ns)),
		"BottomLeftLatitude": float(root.findtext("ns:dataset/ns:GeoInfo/ns:BottomLeftLatitude", namespaces=ns)),
		"TopRightLongitude": float(root.findtext("ns:dataset/ns:GeoInfo/ns:TopRightLongitude", namespaces=ns)),
		"TopRightLatitude": float(root.findtext("ns:dataset/ns:GeoInfo/ns:TopRightLatitude", namespaces=ns)),
		"nx": 67,
		"ny": 120,
		"resolution": 0.03,
	}

	content_node = root.find("ns:dataset/ns:Resource/ns:Content", namespaces=ns)
	if content_node is None or not content_node.text:
		raise ValueError("Content node is missing in the XML file.")

	raw = content_node.text.strip().replace("\n", ",")
	tokens = [t.strip() for t in raw.split(",") if t.strip() != ""]
	values = []
	for t in tokens:
		try:
			values.append(float(t))
		except ValueError:
			continue
	expected = meta["nx"] * meta["ny"]
	if len(values) != expected:
		raise ValueError(f"Parsed values count {len(values)} != expected {expected}")
	grid = np.array(values, dtype=float).reshape(meta["ny"], meta["nx"])  # (lat, lon)
	return meta, grid


def build_lon_lat(meta: dict) -> Tuple[np.ndarray, np.ndarray]:
	lons = meta["BottomLeftLongitude"] + np.arange(meta["nx"]) * meta["resolution"]
	lats = meta["BottomLeftLatitude"] + np.arange(meta["ny"]) * meta["resolution"]
	return lons, lats


# ---------------------------
# Dataset builders
# ---------------------------
def make_datasets(meta: dict, grid: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
	lons, lats = build_lon_lat(meta)
	lon_grid, lat_grid = np.meshgrid(lons, lats)

	values = grid.flatten()
	lon_flat = lon_grid.flatten()
	lat_flat = lat_grid.flatten()

	labels = (values != -999.0).astype(int)
	cls_df = pd.DataFrame({
		"longitude": lon_flat,
		"latitude": lat_flat,
		"label": labels,
	})

	valid_mask = values != -999.0
	reg_df = pd.DataFrame({
		"longitude": lon_flat[valid_mask],
		"latitude": lat_flat[valid_mask],
		"value": values[valid_mask],
	})
	return cls_df, reg_df


# ---------------------------
# Classification: DT with tuned threshold
# ---------------------------
def train_classifier_dt_with_threshold(cls_df: pd.DataFrame, random_state: int = 4):
	X = cls_df[["longitude", "latitude"]].values
	y = cls_df["label"].values
	X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.15, random_state=random_state, stratify=y)
	X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.1764706, random_state=random_state, stratify=y_tmp)

	# Increased tree capacity for a sharper boundary
	clf = DecisionTreeClassifier(random_state=random_state, max_depth=40, class_weight="balanced")
	clf.fit(X_tr, y_tr)

	# tune probability threshold on validation to maximize accuracy
	y_prob_val = clf.predict_proba(X_val)[:, 1]
	cand = np.linspace(0.2, 0.8, 25)
	best_th, best_acc = 0.5, -1.0
	for th in cand:
		acc = float(np.mean((y_prob_val >= th).astype(int) == y_val))
		if acc > best_acc:
			best_acc, best_th = acc, float(th)

	# evaluate on test with tuned threshold
	y_prob_te = clf.predict_proba(X_te)[:, 1]
	y_pred_te = (y_prob_te >= best_th).astype(int)
	cm = confusion_matrix(y_te, y_pred_te)
	acc_te = float(np.mean(y_pred_te == y_te))
	metrics = {
		"accuracy_val": float(best_acc),
		"accuracy_test": float(acc_te),
		"best_threshold": float(best_th),
		"n_train": int(len(y_tr)),
		"n_val": int(len(y_val)),
		"n_test": int(len(y_te)),
		"pos_rate_train": float(np.mean(y_tr)),
		"pos_rate_val": float(np.mean(y_val)),
		"pos_rate_test": float(np.mean(y_te)),
	}
	return clf, metrics, (X_tr, y_tr, X_val, y_val, X_te, y_te)


def plot_classifier_decision_boundary(clf: DecisionTreeClassifier, cls_df: pd.DataFrame, out_path: str,
									  threshold: float, resolution: int = 500, expand: float = 0.1,
									  figsize=(6, 8), point_size: int = 5, show_points: bool = True):
	"""Plot ONLY the tuned-threshold boundary for a probabilistic classifier.

	Points are colored by TN/FP/FN/TP relative to true labels.
	"""
	X = cls_df[["longitude", "latitude"]].to_numpy()
	y = cls_df["label"].to_numpy()
	xmin, xmax = X[:, 0].min(), X[:, 0].max()
	ymin, ymax = X[:, 1].min(), X[:, 1].max()
	xspan, yspan = xmax - xmin, ymax - ymin
	xmin_e = xmin - expand * xspan; xmax_e = xmax + expand * xspan
	ymin_e = ymin - expand * yspan; ymax_e = ymax + expand * yspan

	gx = np.linspace(xmin_e, xmax_e, resolution)
	gy = np.linspace(ymin_e, ymax_e, resolution)
	GX, GY = np.meshgrid(gx, gy)
	grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
	prob = clf.predict_proba(grid_pts)[:, 1].reshape(GX.shape)

	plt.figure(figsize=figsize)
	plt.contour(GX, GY, prob, levels=[threshold], colors='k', linewidths=2)

	if show_points:
		prob_pts = clf.predict_proba(X)[:, 1]
		y_pred = (prob_pts >= threshold).astype(int)
		tn = (y == 0) & (y_pred == 0)
		fp = (y == 0) & (y_pred == 1)
		fn = (y == 1) & (y_pred == 0)
		tp = (y == 1) & (y_pred == 1)
		if tn.any(): plt.scatter(X[tn, 0], X[tn, 1], s=point_size, c="#ABABAB", label='TN', alpha=0.2)
		if fp.any(): plt.scatter(X[fp, 0], X[fp, 1], s=point_size, c="#F93838", label='FP', alpha=0.9)
		if fn.any(): plt.scatter(X[fn, 0], X[fn, 1], s=point_size, c="#6D6BB3", label='FN', alpha=0.9)
		if tp.any(): plt.scatter(X[tp, 0], X[tp, 1], s=point_size, c="#259D43", label='TP', alpha=0.5)

	plt.xlim(xmin_e, xmax_e); plt.ylim(ymin_e, ymax_e)
	plt.xlabel('Longitude'); plt.ylabel('Latitude')
	plt.title(f'Classifier Decision Boundary (p={threshold:.2f})')
	handles, labels = plt.gca().get_legend_handles_labels()
	handles.append(Line2D([0], [0], color='k', lw=2))
	labels.append(f'Boundary p={threshold:.2f}')
	ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
	plt.legend(handles, labels, loc='upper left', frameon=True, framealpha=0.9)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def save_confusion_matrix(cm: np.ndarray, out_path: str):
	fig, ax = plt.subplots(figsize=(4, 4))
	im = ax.imshow(cm, cmap='Blues')
	ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred 0', 'Pred 1'])
	ax.set_yticks([0, 1]); ax.set_yticklabels(['True 0', 'True 1'])
	for i in range(2):
		for j in range(2):
			ax.text(j, i, str(int(cm[i, j])), ha='center', va='center', color='black', fontsize=11)
	ax.set_title('Confusion Matrix')
	fig.tight_layout()
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def plot_tp_regression_overlay(clf: DecisionTreeClassifier, reg_model: Dict[str, np.ndarray], cls_df: pd.DataFrame,
							   out_path: str, threshold: float, resolution: int = 500, expand: float = 0.1,
							   figsize=(6, 8), point_size: int = 10, cmap: str = 'turbo', vmin: float | None = None,
							   vmax: float | None = None):
	"""Decision boundary plot where TP points are colored by regression prediction (hotter = lighter).

	- Draw only the tuned-threshold boundary.
	- Render a faint background of all points in light gray for context.
	- Overlay TP points (true=1, pred=1) with colors from 'hot' colormap using R(x) predictions.
	"""
	X = cls_df[["longitude", "latitude"]].to_numpy()
	y = cls_df["label"].to_numpy()
	xmin, xmax = X[:, 0].min(), X[:, 0].max()
	ymin, ymax = X[:, 1].min(), X[:, 1].max()
	xspan, yspan = xmax - xmin, ymax - ymin
	xmin_e = xmin - expand * xspan; xmax_e = xmax + expand * xspan
	ymin_e = ymin - expand * yspan; ymax_e = ymax + expand * yspan

	gx = np.linspace(xmin_e, xmax_e, resolution)
	gy = np.linspace(ymin_e, ymax_e, resolution)
	GX, GY = np.meshgrid(gx, gy)
	grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
	prob = clf.predict_proba(grid_pts)[:, 1].reshape(GX.shape)

	# Classification outcomes
	prob_pts = clf.predict_proba(X)[:, 1]
	y_pred = (prob_pts >= threshold).astype(int)
	tp_mask = (y == 1) & (y_pred == 1)

	plt.figure(figsize=figsize)
	# boundary at tuned threshold
	plt.contour(GX, GY, prob, levels=[threshold], colors='k', linewidths=2)
	# faint background
	plt.scatter(X[:, 0], X[:, 1], s=4, c="#BBBBBB", alpha=0.15, label='All points (context)')

	# TP overlay colored by regression prediction
	if np.any(tp_mask):
		y_tp_pred = nn_regression_predict(reg_model, X[tp_mask])
		# use provided vmin/vmax to match piecewise heatmap scale if available
		sc = plt.scatter(X[tp_mask, 0], X[tp_mask, 1], s=point_size, c=y_tp_pred, cmap=cmap,
						 vmin=vmin, vmax=vmax, alpha=0.9, edgecolors='none')
		cbar = plt.colorbar(sc)
		cbar.set_label('Predicted Temp (°C)')
	else:
		print("Warning: No TP points to color in TP regression overlay plot.")

	plt.xlim(xmin_e, xmax_e); plt.ylim(ymin_e, ymax_e)
	plt.xlabel('Longitude'); plt.ylabel('Latitude')
	plt.title(f'TP Regression Overlay (boundary p={threshold:.2f})')
	# legend entry for boundary and context
	from matplotlib.lines import Line2D
	handles, labels = plt.gca().get_legend_handles_labels()
	handles.append(Line2D([0], [0], color='k', lw=2))
	labels.append(f'Boundary p={threshold:.2f}')
	ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
	plt.legend(handles, labels, loc='upper left', frameon=True, framealpha=0.9)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_tp_abs_error_overlay(clf: DecisionTreeClassifier, reg_model: Dict[str, np.ndarray], cls_df: pd.DataFrame,
							  y_true_cls: np.ndarray, out_path: str, threshold: float, resolution: int = 500,
							  expand: float = 0.1, figsize=(6, 8), point_size: int = 10):
	"""Decision boundary plot where TP points are colored by absolute error |y_true - R(x)|.

	Color map: white (0) -> red (max error), with a colorbar labeled 'Absolute Error (°C)'.
	Keeps the same boundary and faint background for consistency.
	"""
	X = cls_df[["longitude", "latitude"]].to_numpy()
	y = cls_df["label"].to_numpy()
	xmin, xmax = X[:, 0].min(), X[:, 0].max()
	ymin, ymax = X[:, 1].min(), X[:, 1].max()
	xspan, yspan = xmax - xmin, ymax - ymin
	xmin_e = xmin - expand * xspan; xmax_e = xmax + expand * xspan
	ymin_e = ymin - expand * yspan; ymax_e = ymax + expand * yspan

	gx = np.linspace(xmin_e, xmax_e, resolution)
	gy = np.linspace(ymin_e, ymax_e, resolution)
	GX, GY = np.meshgrid(gx, gy)
	grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
	prob = clf.predict_proba(grid_pts)[:, 1].reshape(GX.shape)

	# outcomes and TP mask
	prob_pts = clf.predict_proba(X)[:, 1]
	y_pred = (prob_pts >= threshold).astype(int)
	tp_mask = (y == 1) & (y_pred == 1)

	plt.figure(figsize=figsize)
	# boundary at tuned threshold
	plt.contour(GX, GY, prob, levels=[threshold], colors='k', linewidths=2)
	# faint background
	plt.scatter(X[:, 0], X[:, 1], s=4, c="#BBBBBB", alpha=0.15, label='All points (context)')

	# TP overlay colored by absolute error
	if np.any(tp_mask):
		y_tp_pred = nn_regression_predict(reg_model, X[tp_mask])
		y_tp_true = y_true_cls[tp_mask]
		errs = np.abs(y_tp_true - y_tp_pred)
		vmax = float(np.max(errs)) if errs.size > 0 else 1.0
		sc = plt.scatter(X[tp_mask, 0], X[tp_mask, 1], s=point_size, c=errs, cmap='Reds', vmin=0.0, vmax=vmax, alpha=0.95, edgecolors='none')
		cbar = plt.colorbar(sc)
		cbar.set_label('Absolute Error (°C)')
	else:
		print("Warning: No TP points to color in TP abs-error overlay plot.")

	plt.xlim(xmin_e, xmax_e); plt.ylim(ymin_e, ymax_e)
	plt.xlabel('Longitude'); plt.ylabel('Latitude')
	plt.title(f'TP Absolute Error Overlay (boundary p={threshold:.2f})')
	from matplotlib.lines import Line2D
	handles, labels = plt.gca().get_legend_handles_labels()
	handles.append(Line2D([0], [0], color='k', lw=2))
	labels.append(f'Boundary p={threshold:.2f}')
	ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
	plt.legend(handles, labels, loc='upper left', frameon=True, framealpha=0.9)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# ---------------------------
# Regression: Neural Network (MLP) fit + predict
# ---------------------------
def nn_regression_fit(X: np.ndarray, y: np.ndarray, random_state: int = 4) -> Dict[str, object]:
	scaler = StandardScaler()
	X_s = scaler.fit_transform(X)
	regr = MLPRegressor(
		hidden_layer_sizes=(64, 64),
		activation="relu",
		random_state=random_state,
		max_iter=10000,  # more epochs/iterations
		verbose=True,   # print training progress
		early_stopping=True,
		tol=1e-7,        # lower tolerance than default 1e-4
		n_iter_no_change=30,
		validation_fraction=0.15,
	)
	regr.fit(X_s, y)
	return {"scaler": scaler, "regr": regr}


def nn_regression_predict(model: Dict[str, object], X: np.ndarray) -> np.ndarray:
	scaler: StandardScaler = model["scaler"]
	regr: MLPRegressor = model["regr"]
	X_s = scaler.transform(X)
	return regr.predict(X_s)


# ---------------------------
# Piecewise combiner
# ---------------------------
def piecewise_predict(clf: DecisionTreeClassifier, reg_model: Dict[str, np.ndarray], X_all: np.ndarray, threshold: float) -> np.ndarray:
	"""Return h(x) for each x in X_all: regression where clf prob>=threshold else -999."""
	p = clf.predict_proba(X_all)[:, 1]
	mask = p >= threshold
	y_hat = np.full(len(X_all), -999.0, dtype=float)
	if mask.any():
		y_hat[mask] = nn_regression_predict(reg_model, X_all[mask])
	return y_hat


def plot_piecewise_heatmap(meta: dict, lons: np.ndarray, lats: np.ndarray, h_vals: np.ndarray, out_path: str) -> tuple[float, float]:
	"""Plot heatmap of the piecewise function h over the grid extents.

	h_vals is a flat array ordered row-major by (lats, lons) matching meshgrid.
	"""
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	H = h_vals.reshape(len(lats), len(lons))
	plt.figure(figsize=(6, 8))
	valid_mask = (H != -999)
	if np.any(valid_mask):
		vmin = float(np.nanmin(H[valid_mask]))
		vmax = float(np.nanmax(H[valid_mask]))
	else:
		# Fallback to a safe range if nothing valid (rare)
		vmin, vmax = 0.0, 1.0
	im = plt.imshow(
		H,
		origin="lower",
		extent=[lons.min(), lons.max(), lats.min(), lats.max()],
		aspect="auto",
		cmap="coolwarm",
		vmin=vmin,
		vmax=vmax,
	)
	plt.colorbar(im, label="Temperature (°C)")
	plt.xlabel("Longitude"); plt.ylabel("Latitude")
	plt.title("Piecewise Temperature $h(x)$")
	plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
	return vmin, vmax


def plot_regression_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
	"""Scatter plot of true vs predicted temperatures with identity line."""
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.figure(figsize=(5, 5))
	plt.scatter(y_true, y_pred, s=8, alpha=0.6)
	vmin = float(np.nanmin([y_true.min(), y_pred.min()]))
	vmax = float(np.nanmax([y_true.max(), y_pred.max()]))
	plt.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1)
	plt.xlabel('True Temperature (°C)')
	plt.ylabel('Predicted Temperature (°C)')
	plt.title('Regression: True vs Predicted')
	plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def main():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	xml_path = os.path.join(BASE_DIR, "O-A0038-003.xml")
	fig_dir = os.path.join(BASE_DIR, "figures")
	os.makedirs(fig_dir, exist_ok=True)

	# 1) Load data and build datasets
	meta, grid = parse_xml(xml_path)
	cls_df, reg_df = make_datasets(meta, grid)
	print("Datasets built:", len(cls_df), "classification points;", len(reg_df), "regression points")

	# 2) Train classifier (DT with tuned threshold)
	clf, cls_metrics, (X_tr, y_tr, X_val, y_val, X_te, y_te) = train_classifier_dt_with_threshold(cls_df, random_state=4)
	print("Classifier (DT) with tuned threshold:")
	print(f"- Split: train {cls_metrics['n_train']} / val {cls_metrics['n_val']} / test {cls_metrics['n_test']}")
	print(f"- Class ratio: train {cls_metrics['pos_rate_train']:.3f}, val {cls_metrics['pos_rate_val']:.3f}, test {cls_metrics['pos_rate_test']:.3f}")
	print(f"- Best threshold (val): {cls_metrics['best_threshold']:.3f}")
	print(f"- Val accuracy: {cls_metrics['accuracy_val']:.4f}")
	print(f"- Test accuracy: {cls_metrics['accuracy_test']:.4f}")

	# Confusion matrix on test
	y_prob_te = clf.predict_proba(X_te)[:, 1]
	y_pred_te = (y_prob_te >= cls_metrics['best_threshold']).astype(int)
	cm = confusion_matrix(y_te, y_pred_te)
	cm_path = os.path.join(fig_dir, "confusion_matrix_piecewise.png")
	save_confusion_matrix(cm, cm_path)
	print("Saved:", cm_path)

	# Decision boundary plot (best-p only)
	db_path = os.path.join(fig_dir, "decision_boundary_piecewise.png")
	plot_classifier_decision_boundary(clf, cls_df, db_path, threshold=cls_metrics['best_threshold'], resolution=500, expand=0, figsize=(6,8), point_size=5, show_points=True)
	print("Saved:", db_path)

	# 3) Train regression (Neural Network on valid points)
	X_reg = reg_df[["longitude", "latitude"]].to_numpy()
	y_reg = reg_df["value"].to_numpy()
	reg_model = nn_regression_fit(X_reg, y_reg, random_state=4)
	print("Trained NN regression (MLP 64-64)")
	# Regression scatter: predictions on all valid points
	y_reg_pred = nn_regression_predict(reg_model, X_reg)
	reg_scatter_path = os.path.join(fig_dir, "regression_true_vs_pred_NN.png")
	plot_regression_true_vs_pred(y_reg, y_reg_pred, reg_scatter_path)
	print("Saved:", reg_scatter_path)

	# (Overlay will be generated after computing h_vals to share vmin/vmax)

	# 4) Piecewise combine over the full grid
	X_all = cls_df[["longitude", "latitude"]].to_numpy()
	h_vals = piecewise_predict(clf, reg_model, X_all, threshold=cls_metrics['best_threshold'])
	n_valid = int(np.sum(h_vals != -999))
	n_total = int(h_vals.size)
	print(f"Piecewise predictions: valid {n_valid} / total {n_total} = {n_valid/n_total:.3f}")

	# 4b) Build true values array aligned with cls_df for absolute error coloring (invalid as NaN)
	# Reconstruct the full grid true values matching cls_df order (row-major by lat,lon)
	true_vals_full = grid.flatten()
	y_true_full = true_vals_full.copy()
	y_true_full[true_vals_full == -999.0] = np.nan  # NaN where invalid

	# 5) Plot the piecewise heatmap
	lons, lats = build_lon_lat(meta)
	h_path = os.path.join(fig_dir, "piecewise_temperature.png")
	vmin, vmax = plot_piecewise_heatmap(meta, lons, lats, h_vals, h_path)
	print("Saved:", h_path)

	# TP regression overlay decision map (requires reg_model) with same color scaling as piecewise heatmap
	db_tp_path = os.path.join(fig_dir, "decision_boundary_tp_overlay.png")
	plot_tp_regression_overlay(clf, reg_model=reg_model, cls_df=cls_df, out_path=db_tp_path, threshold=cls_metrics['best_threshold'], resolution=500, expand=0, figsize=(6,8), point_size=12, cmap='coolwarm', vmin=vmin, vmax=vmax)
	print("Saved:", db_tp_path)

	# 5b) TP absolute error overlay (white=0, red=max)
	db_tp_err_path = os.path.join(fig_dir, "decision_boundary_tp_abs_error.png")
	plot_tp_abs_error_overlay(clf, reg_model=reg_model, cls_df=cls_df, y_true_cls=y_true_full, out_path=db_tp_err_path, threshold=cls_metrics['best_threshold'], resolution=500, expand=0, figsize=(6,8), point_size=12)
	print("Saved:", db_tp_err_path)

	# 6) Brief explanation (print)
	print("Explanation: h(x) uses the classifier DT boundary to mask invalid zones (predict 0).")
	print("Where C(x)=1 (prob >= best threshold), we plug in the neural network regression R(x); otherwise output -999.")


if __name__ == "__main__":
	main()

