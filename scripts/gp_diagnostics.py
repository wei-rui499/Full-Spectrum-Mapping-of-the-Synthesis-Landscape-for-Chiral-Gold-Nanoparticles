"""
Gaussian Process regression diagnostics: LOO cross-validation, learning curves, SHAP dependence.

Outputs:
  figures/SI_GP_LOO_scatter.pdf
  figures/SI_GP_learning_curve.pdf
  figures/SI_GP_SHAP_dependence.pdf
"""
import runtime  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, KFold, learning_curve
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
import shap
import style as js
from pipeline import load_data, run_fpca, cluster_kmeans, RANDOM_STATE

js.apply_style()
PARAMS = ["CTAB", "Au", "AA", "GSH"]
PARAM_COLORS = [js.PALETTE_30[2], js.PALETTE_30[15], js.PALETTE_30[8], js.PALETTE_30[24]]
SHAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "shap_cmap", [js.PALETTE_30[2], "#FFFFFF", js.PALETTE_30[23]], N=256)

# ── Load ────────────────────────────────────────────────
wavelengths, spectra, params_df, sids = load_data()
fpc_scores = run_fpca(wavelengths, spectra)
labels, _ = cluster_kmeans(fpc_scores)
X = params_df[PARAMS].values.astype(float)
y = params_df["g_factor"].values.astype(float)
cluster_colors_arr = [js.CLUSTER_COLORS_4[l] for l in labels]

# ── Train GP ────────────────────────────────────────────
kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)
pipe = Pipeline([("scaler", StandardScaler()),
    ("model", GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20,
                                        alpha=1e-6, normalize_y=True,
                                        random_state=RANDOM_STATE))])

# LOO-CV
loo_pred = np.zeros(len(y))
for tr, te in LeaveOneOut().split(X):
    p = clone(pipe); p.fit(X[tr], y[tr])
    loo_pred[te] = p.predict(X[te])
loo_r2 = r2_score(y, loo_pred)
loo_rmse = root_mean_squared_error(y, loo_pred)
loo_mae = mean_absolute_error(y, loo_pred)
print(f"GP LOO: R²={loo_r2:.4f}, RMSE={loo_rmse:.4f}, MAE={loo_mae:.4f}")

# ── SI Fig S11a: LOO scatter ────────────────────────────
fig, ax = plt.subplots(figsize=(js.SINGLE_COL, js.SINGLE_COL))
ax.scatter(y, loo_pred, c=cluster_colors_arr, s=30, alpha=0.85,
           edgecolors="white", lw=0.3, zorder=3)
ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("True |g-factor|"); ax.set_ylabel("Predicted |g-factor| (LOO)")
ax.text(0.05, 0.92, f"LOO R² = {loo_r2:.3f}\nRMSE = {loo_rmse:.4f}\nMAE = {loo_mae:.4f}",
        transform=ax.transAxes, fontsize=js.FONTSIZE_TICK, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9, lw=0.5))
js.save_fig(fig, "SI_GP_LOO_scatter")
print("  -> figures/SI_GP_LOO_scatter.pdf")

# ── SI Fig S11b: Learning curve ─────────────────────────
fig, ax = plt.subplots(figsize=(js.SINGLE_COL, js.SINGLE_COL * 0.85))
tr_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(
    pipe, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    train_sizes=tr_sizes, scoring="r2", n_jobs=1)
tm, ts = train_scores.mean(axis=1), train_scores.std(axis=1)
cm, cs = test_scores.mean(axis=1), test_scores.std(axis=1)
ax.fill_between(train_sizes, tm - ts, tm + ts, alpha=0.15, color=js.PALETTE_30[2])
ax.fill_between(train_sizes, cm - cs, cm + cs, alpha=0.15, color=js.PALETTE_30[25])
ax.plot(train_sizes, tm, "o-", color=js.PALETTE_30[2], lw=1.2, ms=4, label="Train R²")
ax.plot(train_sizes, cm, "o--", color=js.PALETTE_30[25], lw=1.2, ms=4, label="CV R²")
ax.axhline(loo_r2, color="grey", lw=0.8, ls=":", label=f"LOO R² = {loo_r2:.3f}")
ax.set_xlabel("Training Size"); ax.set_ylabel("R² Score")
ax.legend(fontsize=js.FONTSIZE_LEGEND, frameon=False)
js.save_fig(fig, "SI_GP_learning_curve")
print("  -> figures/SI_GP_learning_curve.pdf")

# ── SI Fig S12: SHAP dependence ─────────────────────────
pipe.fit(X, y)
X_scaled = pipe["scaler"].transform(X)
model = pipe["model"]
background = shap.kmeans(X_scaled, 10)
explainer = shap.KernelExplainer(model.predict, background)
sv = explainer.shap_values(X_scaled, nsamples=200)
sv = np.asarray(sv)
if sv.ndim == 3:
    sv = sv[:, :, 0] if sv.shape[0] == X_scaled.shape[0] else sv[0]
if sv.ndim != 2:
    raise ValueError(f"Unexpected SHAP value shape: {sv.shape}")

fig, axes = plt.subplots(2, 2, figsize=(js.DOUBLE_COL * 0.85, js.DOUBLE_COL * 0.75))
for ax, pi, pname in zip(axes.flat, range(4), ["CTAB", "HAuCl₄", "AA", "GSH"]):
    sort_idx = np.argsort(X[:, pi])
    ax.scatter(X[sort_idx, pi], sv[sort_idx, pi], c=X[sort_idx, pi],
               cmap=SHAP_CMAP, s=25, alpha=0.85, edgecolors="none")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    unit = "µM" if pname == "GSH" else "mM"
    ax.set_xlabel(f"{pname} ({unit})")
    ax.set_ylabel(f"SHAP value")
fig.suptitle("SHAP Dependence — Gaussian Process Regression", fontweight="bold")
fig.tight_layout()
js.save_fig(fig, "SI_GP_SHAP_dependence")
print("  -> figures/SI_GP_SHAP_dependence.pdf")
print("Done.")
