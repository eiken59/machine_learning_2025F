"""
Minimal temperature data loader + dataset builder + from-scratch QDA (part a)

This module currently includes:
- parse_xml / build_lon_lat / make_datasets
- QDA implementation (qda_fit / qda_predict_proba / qda_predict)

Later steps (b-d) like training protocol, metrics, and decision boundary plots
can be added on top of this clean base.

Notes:
- Grid is 67 (lon) x 120 (lat), resolution 0.03 degrees.
- Longitude increases west->east along each row, then latitude increases south->north by rows.
"""

import os
import xml.etree.ElementTree as ET
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
		# ---------------------------
		# Train/Val/Test 70/15/15 with stratification
		# ---------------------------
		"longitude": lon_flat[valid_mask],
		"latitude": lat_flat[valid_mask],
		"value": values[valid_mask],
	})

	return cls_df, reg_df

# ---------------------------
# QDA
# ---------------------------
def _log_gaussian(X: np.ndarray, mu: np.ndarray, Sigma_inv: np.ndarray, log_det: float) -> np.ndarray:
	"""Vectorized log N(X | mu, Sigma) for rows in X.

	Returns shape (n,) of log densities.
	"""
	diff = X - mu  # (n,d)
	quad = np.einsum("ni,ij,nj->n", diff, Sigma_inv, diff)  # (n,)
	d = X.shape[1]
	return -0.5 * (d * np.log(2 * np.pi) + log_det + quad)


def qda_fit(X: np.ndarray, y: np.ndarray, reg: float = 1e-6) -> Dict[str, np.ndarray]:
	"""Fit binary QDA with class-specific covariances.

	Parameters
	----------
	X : (n, d) array
	y : (n,) array of {0,1}
	reg : float, small ridge added to diagonals of each covariance for stability

	Returns
	-------
	params : dict with keys {phi, mu0, mu1, Sigma0, Sigma1, inv0, inv1, logdet0, logdet1}

	Notes
	-----
	- QDA models p(x|y=k) as N(mu_k, Sigma_k) with separate covariances.
	- Posteriors computed via Bayes rule using class priors phi and 1-phi.
	"""
	if X.ndim != 2:
		raise ValueError("X must be 2D array")
	classes = np.unique(y)
	if set(classes) - {0, 1}:
		raise ValueError("y must be binary {0,1}")

	phi = float(np.mean(y))
	X0 = X[y == 0]
	X1 = X[y == 1]
	mu0 = X0.mean(axis=0)
	mu1 = X1.mean(axis=0)
	# Unbiased covariance (divide by n_k) for ML estimate; ridge regularization
	S0 = np.cov(X0, rowvar=False, bias=True)
	S1 = np.cov(X1, rowvar=False, bias=True)
	S0 = S0 + reg * np.eye(S0.shape[0])
	S1 = S1 + reg * np.eye(S1.shape[0])
	inv0 = np.linalg.inv(S0)
	inv1 = np.linalg.inv(S1)
	logdet0 = float(np.log(np.linalg.det(S0) + 1e-18))
	logdet1 = float(np.log(np.linalg.det(S1) + 1e-18))
	return {
		"phi": phi,
		"mu0": mu0,
		"mu1": mu1,
		"Sigma0": S0,
		"Sigma1": S1,
		"inv0": inv0,
		"inv1": inv1,
		"logdet0": logdet0,
		"logdet1": logdet1,
	}


def qda_predict_proba(params: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
	"""Return p(y=1 | x) for each row of X using QDA posteriors.

	We compute unnormalized log-joint:
		l0 = log(1-phi) + log N(x | mu0, S0)
		l1 = log(phi)   + log N(x | mu1, S1)
	Then normalize with log-sum-exp for stability.
	"""
	phi = params["phi"]
	mu0 = params["mu0"]; mu1 = params["mu1"]
	inv0 = params["inv0"]; inv1 = params["inv1"]
	logdet0 = params["logdet0"]; logdet1 = params["logdet1"]
	log_p0 = np.log(1 - phi + 1e-18) + _log_gaussian(X, mu0, inv0, logdet0)
	log_p1 = np.log(phi + 1e-18)     + _log_gaussian(X, mu1, inv1, logdet1)
	# log-sum-exp
	M = np.maximum(log_p0, log_p1)
	den = M + np.log(np.exp(log_p0 - M) + np.exp(log_p1 - M))
	return np.exp(log_p1 - den)


def qda_predict(params: Dict[str, np.ndarray], X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
	"""Return hard class predictions (0/1) using threshold on p(y=1|x)."""
	probs = qda_predict_proba(params, X)
	return (probs >= threshold).astype(int)


# ---------------------------
# training + accuracy reporting
# ---------------------------
def _stratified_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 4):
	"""Simple stratified split implemented with numpy (no sklearn required)."""
	rng = np.random.default_rng(random_state)
	idx0 = np.where(y == 0)[0]
	idx1 = np.where(y == 1)[0]
	rng.shuffle(idx0); rng.shuffle(idx1)
	n0 = len(idx0); n1 = len(idx1)
	n0_te = int(round(test_size * n0)); n1_te = int(round(test_size * n1))
	te_idx = np.concatenate([idx0[:n0_te], idx1[:n1_te]])
	tr_idx = np.concatenate([idx0[n0_te:], idx1[n1_te:]])
	# Shuffle combined to avoid class blocks
	rng.shuffle(tr_idx); rng.shuffle(te_idx)
	return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Compute classification accuracy."""
	return float(np.mean(y_true == y_pred))


def qda_train_and_evaluate_holdout(cls_df: 'pd.DataFrame', test_size: float = 0.3, random_state: int = 4, reg: float = 1e-6, threshold: float = 0.5):
	"""Train QDA on a stratified holdout split and report accuracy.

	Measurement protocol:
	- Split the classification dataset (longitude, latitude -> label) into
	  train/test with stratification (preserving class ratio), test_size proportion.
	- Fit QDA parameters on the train split only.
	- Predict on the held-out test split and compute accuracy.

	Returns a tuple: (metrics dict, params)
	metrics has keys: accuracy, n_train, n_test, pos_rate_train, pos_rate_test
	"""
	X = cls_df[["longitude", "latitude"]].to_numpy()
	y = cls_df["label"].to_numpy()
	X_tr, X_te, y_tr, y_te = _stratified_train_test_split(X, y, test_size=test_size, random_state=random_state)
	params = qda_fit(X_tr, y_tr, reg=reg)
	y_pred = qda_predict(params, X_te, threshold=threshold)
	acc = _accuracy(y_te, y_pred)
	metrics = {
		"accuracy": acc,
		"n_train": int(len(y_tr)),
		"n_test": int(len(y_te)),
		"pos_rate_train": float(np.mean(y_tr)),
		"pos_rate_test": float(np.mean(y_te)),
		"test_size": float(test_size),
	}
	return metrics, params


def stratified_train_val_test_split(X: np.ndarray, y: np.ndarray, random_state: int = 4):
	"""Return (X_tr, y_tr), (X_val, y_val), (X_te, y_te) with approx 70/15/15 stratified.

	Approach: First take 15% test per class; from remaining per-class, take 15/85≈17.647% for val.
	"""
	rng = np.random.default_rng(random_state)
	idx0 = np.where(y == 0)[0]
	idx1 = np.where(y == 1)[0]
	rng.shuffle(idx0); rng.shuffle(idx1)
	n0 = len(idx0); n1 = len(idx1)
	n0_te = int(round(0.15 * n0)); n1_te = int(round(0.15 * n1))
	te0, rem0 = idx0[:n0_te], idx0[n0_te:]
	te1, rem1 = idx1[:n1_te], idx1[n1_te:]
	# val proportion of remaining is 0.1764706 ~ 3/17 ≈ 15% absolute
	n0_val = int(round(0.1764706 * len(rem0)))
	n1_val = int(round(0.1764706 * len(rem1)))
	val0, tr0 = rem0[:n0_val], rem0[n0_val:]
	val1, tr1 = rem1[:n1_val], rem1[n1_val:]
	tr_idx = np.concatenate([tr0, tr1])
	val_idx = np.concatenate([val0, val1])
	te_idx = np.concatenate([te0, te1])
	rng.shuffle(tr_idx); rng.shuffle(val_idx); rng.shuffle(te_idx)
	return (X[tr_idx], y[tr_idx]), (X[val_idx], y[val_idx]), (X[te_idx], y[te_idx])


def qda_train_val_test_with_threshold(cls_df: 'pd.DataFrame', random_state: int = 4, reg: float = 1e-6):
	"""Train on 70%, select threshold on 15% validation (maximize accuracy), report on 15% test.

	Returns: (metrics dict, params, best_threshold)
	"""
	X = cls_df[["longitude", "latitude"]].to_numpy()
	y = cls_df["label"].to_numpy()
	(X_tr, y_tr), (X_val, y_val), (X_te, y_te) = stratified_train_val_test_split(X, y, random_state=random_state)
	params = qda_fit(X_tr, y_tr, reg=reg)
	# tune threshold on validation by maximizing accuracy
	prob_val = qda_predict_proba(params, X_val)
	candidate_th = np.linspace(0.2, 0.8, 25)
	best_th, best_acc = 0.5, -1.0
	for th in candidate_th:
		acc = _accuracy(y_val, (prob_val >= th).astype(int))
		if acc > best_acc:
			best_acc, best_th = acc, float(th)
	prob_te = qda_predict_proba(params, X_te)
	y_pred = (prob_te >= best_th).astype(int)
	acc_te = _accuracy(y_te, y_pred)
	metrics = {
		"accuracy_test": float(acc_te),
		"accuracy_val": float(best_acc),
		"best_threshold": float(best_th),
		"n_train": int(len(y_tr)),
		"n_val": int(len(y_val)),
		"n_test": int(len(y_te)),
		"pos_rate_train": float(np.mean(y_tr)),
		"pos_rate_val": float(np.mean(y_val)),
		"pos_rate_test": float(np.mean(y_te)),
	}
	return metrics, params, best_th


def plot_qda_decision_boundary(params: Dict[str, np.ndarray], cls_df: 'pd.DataFrame', out_path: str,
							   resolution: int = 400, expand: float = 0.2, figsize=(6,6), point_size=6,
							   show_points: bool = True, threshold: float = 0.5):
	"""Plot only the QDA decision boundary at the provided threshold and optionally TN/FP/FN/TP points.

	Args:
		params: QDA parameters from qda_fit.
		cls_df: DataFrame with columns [longitude, latitude, label].
		out_path: PNG save path.
		resolution: grid resolution for contour.
		expand: fractional padding for axes limits.
		figsize: figure size.
		point_size: scatter marker size.
		show_points: if True, draw points; else draw boundary only.
		threshold: classification threshold for coloring points (default 0.5). Backward compatible.
	"""
	X = cls_df[["longitude", "latitude"]].to_numpy()
	y = cls_df["label"].to_numpy()
	xmin, xmax = X[:,0].min(), X[:,0].max()
	ymin, ymax = X[:,1].min(), X[:,1].max()
	xspan, yspan = xmax - xmin, ymax - ymin
	xmin_e = xmin - expand * xspan; xmax_e = xmax + expand * xspan
	ymin_e = ymin - expand * yspan; ymax_e = ymax + expand * yspan
	gx = np.linspace(xmin_e, xmax_e, resolution)
	gy = np.linspace(ymin_e, ymax_e, resolution)
	GX, GY = np.meshgrid(gx, gy)
	grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
	prob = qda_predict_proba(params, grid_pts).reshape(GX.shape)
	plt.figure(figsize=figsize)
	# draw only the best-threshold boundary
	try:
		th_val = float(threshold)
	except Exception:
		th_val = 0.5
	plt.contour(GX, GY, prob, levels=[th_val], colors='k', linewidths=2)
	# optionally show points colored by outcomes
	if show_points:
		prob_pts = qda_predict_proba(params, X)
		y_pred = (prob_pts >= threshold).astype(int)
		tn = (y == 0) & (y_pred == 0)
		fp = (y == 0) & (y_pred == 1)
		fn = (y == 1) & (y_pred == 0)
		tp = (y == 1) & (y_pred == 1)
		if tn.any():
			plt.scatter(X[tn,0], X[tn,1], s=point_size, c="#ABABAB", label='TN (label = 0, predicted correctly)', alpha=0.2)
		if fp.any():
			plt.scatter(X[fp,0], X[fp,1], s=point_size, c="#F93838", label='FP (label = 0, predicted wrong)', alpha=0.9)
		if fn.any():
			plt.scatter(X[fn,0], X[fn,1], s=point_size, c="#6D6BB3", label='FN (label = 1, predicted wrong)', alpha=0.9)
		if tp.any():
			plt.scatter(X[tp,0], X[tp,1], s=point_size, c="#259D43", label='TP (label = 1, predicted correctly)', alpha=0.5)
	plt.xlim(xmin_e, xmax_e); plt.ylim(ymin_e, ymax_e)
	plt.xlabel('Longitude'); plt.ylabel('Latitude')
	plt.title('QDA Decision Boundary')
	# legend include boundary only
	from matplotlib.lines import Line2D
	handles, labels = plt.gca().get_legend_handles_labels()
	handles.append(Line2D([0],[0], color='k', lw=2))
	labels.append(f'Boundary p={th_val:.2f}')
	ax = plt.gca()
	ax.set_aspect('equal', adjustable='box')
	plt.legend(handles, labels, loc='upper left', frameon=True, framealpha=0.9)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# ---------------------------
# Confusion matrix utilities (binary)
# ---------------------------
def _confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
	"""Return 2x2 confusion matrix [[TN, FP],[FN, TP]] for binary labels {0,1}."""
	y_true = np.asarray(y_true).astype(int)
	y_pred = np.asarray(y_pred).astype(int)
	tn = int(np.sum((y_true == 0) & (y_pred == 0)))
	fp = int(np.sum((y_true == 0) & (y_pred == 1)))
	fn = int(np.sum((y_true == 1) & (y_pred == 0)))
	tp = int(np.sum((y_true == 1) & (y_pred == 1)))
	return np.array([[tn, fp],[fn, tp]], dtype=int)


def plot_confusion_matrix_simple(cm: np.ndarray, out_path: str):
	"""Plot a simple confusion matrix heatmap without external libs.

	cm layout: [[TN, FP], [FN, TP]]
	"""
	fig, ax = plt.subplots(figsize=(4,4))
	im = ax.imshow(cm, cmap='Blues')
	ax.set_xticks([0,1]); ax.set_yticks([0,1])
	ax.set_xticklabels(['Pred 0','Pred 1'])
	ax.set_yticklabels(['True 0','True 1'])
	# annotate
	for i in range(2):
		for j in range(2):
			ax.text(j, i, str(int(cm[i,j])), ha='center', va='center', color='black', fontsize=11)
	ax.set_title('Confusion Matrix')
	fig.tight_layout()
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	fig.savefig(out_path, dpi=150)
	plt.close(fig)

if __name__ == "__main__":
	# Minimal runnable entry for part (c+d): load XML, build dataset, 70/15/15 split,
	# tune threshold on val for accuracy, evaluate on test, and save decision boundary.
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	xml_path = os.path.join(BASE_DIR, "O-A0038-003.xml")
	fig_dir = os.path.join(BASE_DIR, "figures")
	meta, grid = parse_xml(xml_path)
	cls_df, _ = make_datasets(meta, grid)
	metrics, params, best_th = qda_train_val_test_with_threshold(cls_df, random_state=4, reg=1e-6)
	print("QDA 70/15/15 (stratified) with val threshold tuning:")
	print(f"- Split: train {metrics['n_train']} / val {metrics['n_val']} / test {metrics['n_test']}")
	print(f"- Class ratio: train {metrics['pos_rate_train']:.3f}, val {metrics['pos_rate_val']:.3f}, test {metrics['pos_rate_test']:.3f}")
	print(f"- Best threshold (val): {metrics['best_threshold']:.3f}")
	print(f"- Val accuracy: {metrics['accuracy_val']:.4f}")
	print(f"- Test accuracy: {metrics['accuracy_test']:.4f}")
	out_png = os.path.join(fig_dir, "decision_boundary_qda.png")
	# Plot decision boundary and color points by TN/FP/FN/TP using tuned threshold
	plot_qda_decision_boundary(params, cls_df, out_png, resolution=500, expand=0.2, figsize=(6,8), point_size=5, threshold=metrics['best_threshold'])
	print(f"Plotted decision boundary with best p (threshold) = {metrics['best_threshold']:.3f}")
	print("Saved:", out_png)

	# Compute confusion matrix on test set using same split seed and tuned threshold (does not change printed metrics)
	X = cls_df[["longitude", "latitude"]].to_numpy()
	y = cls_df["label"].to_numpy()
	(_, _), (_, _), (X_te, y_te) = stratified_train_val_test_split(X, y, random_state=4)
	y_te_pred = qda_predict(params, X_te, threshold=metrics['best_threshold'])
	cm = _confusion_matrix_binary(y_te, y_te_pred)
	cm_path = os.path.join(fig_dir, "confusion_matrix_qda.png")
	plot_confusion_matrix_simple(cm, cm_path)
	print("Saved:", cm_path)