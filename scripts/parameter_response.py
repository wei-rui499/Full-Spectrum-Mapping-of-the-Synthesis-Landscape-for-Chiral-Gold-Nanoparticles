"""
Gaussian process regression and SHAP analysis of synthesis parameter–|g-factor| relationships.

Outputs:
  figures/Fig3_correlation_matrix.pdf
  figures/Fig3_GP_response_surface.pdf
  figures/Fig3_SHAP_beeswarm.pdf
"""
import runtime  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
import shap
import style as js
from pipeline import load_data, run_fpca, cluster_kmeans, RANDOM_STATE

js.apply_style()
PARAMS = ["CTAB", "Au", "AA", "GSH"]
PARAM_LABELS = ["CTAB (mM)", "HAuCl₄ (mM)", "AA (mM)", "GSH (μM)"]
PARAM_COLORS = [js.PALETTE_30[2], js.PALETTE_30[15], js.PALETTE_30[8], js.PALETTE_30[24]]
SHAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "shap_cmap", [js.PALETTE_30[2], "#FFFFFF", js.PALETTE_30[23]], N=256)

# ── Load data & cluster labels ──────────────────────────
wavelengths, spectra, params, sids = load_data()
fpc_scores = run_fpca(wavelengths, spectra)
labels, _ = cluster_kmeans(fpc_scores)
X = params[PARAMS].values.astype(float)
y = params["g_factor"].values.astype(float)

# ════════════════════════════════════════════════════════
# Fig 3a: Pearson correlation matrix
# ════════════════════════════════════════════════════════
df_corr = params[["CTAB", "Au", "AA", "GSH", "g_factor"]].copy()
df_corr.columns = ["CTAB", "HAuCl₄", "AA", "GSH", "|g|"]
corr = df_corr.corr()
n_c = len(df_corr.columns)

fig, axes = plt.subplots(n_c, n_c, figsize=(js.DOUBLE_COL * 0.7, js.DOUBLE_COL * 0.7))
for i in range(n_c):
    for j in range(n_c):
        ax = axes[i, j]
        if i == j:
            ax.hist(df_corr.iloc[:, i], bins=15, color=js.PALETTE_30[2], alpha=0.7,
                    edgecolor="white", lw=0.3)
            ax.set_xlabel(df_corr.columns[i], fontsize=7)
            ax.set_ylabel("Count" if j == 0 else "", fontsize=7)
        elif j < i:
            xi, xj = df_corr.iloc[:, j].values, df_corr.iloc[:, i].values
            ax.scatter(xi, xj, s=12, alpha=0.6, color=js.PALETTE_30[2],
                       edgecolors="none")
            ax.set_xlabel(df_corr.columns[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(df_corr.columns[i], fontsize=7)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{corr.iloc[i, j]:.2f}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if abs(corr.iloc[i, j]) > 0.3 else "black")
            ax.set_facecolor(js.PALETTE_30[2] if abs(corr.iloc[i, j]) > 0.3 else "#f0f0f0")
fig.tight_layout()
js.save_fig(fig, "Fig3_correlation_matrix")
print("  -> figures/Fig3_correlation_matrix.pdf")

# ════════════════════════════════════════════════════════
# Fig 3b: GP response surfaces (Au vs CTAB/AA/GSH)
# ════════════════════════════════════════════════════════
kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)
gp = Pipeline([("scaler", StandardScaler()),
               ("model", GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20,
                                                   alpha=1e-6, normalize_y=True,
                                                   random_state=RANDOM_STATE))])
gp.fit(X, y)
print(f"GP: LOO R² evaluation (see SI Fig 11 for full diagnostics)")

other_params = [("CTAB", 0), ("AA", 2), ("GSH", 3)]
AU_IDX = 1
fig, axes = plt.subplots(1, 3, figsize=(js.DOUBLE_COL, js.SINGLE_COL * 0.7))
for ax, (pname, pidx) in zip(axes, other_params):
    # Grid
    au_range = np.linspace(X[:, AU_IDX].min(), X[:, AU_IDX].max(), 50)
    other_range = np.linspace(X[:, pidx].min(), X[:, pidx].max(), 50)
    AA_grid, BB_grid = np.meshgrid(au_range, other_range)
    X_grid = np.zeros((AA_grid.size, 4))
    X_grid[:, AU_IDX] = AA_grid.ravel()
    X_grid[:, pidx] = BB_grid.ravel()
    # Fill remaining params with median
    for k in range(4):
        if k not in (AU_IDX, pidx):
            X_grid[:, k] = np.median(X[:, k])
    Z = gp.predict(X_grid).reshape(AA_grid.shape)

    cs = ax.contourf(AA_grid, BB_grid, Z, levels=15, cmap="RdYlBu_r", alpha=0.85)
    ax.scatter(X[:, AU_IDX], X[:, pidx], c=y, cmap="RdYlBu_r", s=20,
               edgecolors="white", lw=0.3, zorder=3)
    ax.set_xlabel("HAuCl₄ (mM)"); ax.set_ylabel(f"{pname} " + ("(mM)" if pname != "GSH" else "(μM)"))
fig.tight_layout()
js.save_fig(fig, "Fig3_GP_response_surface")
print("  -> figures/Fig3_GP_response_surface.pdf")

# ════════════════════════════════════════════════════════
# Fig 3c: SHAP beeswarm
# ════════════════════════════════════════════════════════
X_scaled = gp["scaler"].transform(X)
model = gp["model"]
background = shap.kmeans(X_scaled, 10)
explainer = shap.KernelExplainer(model.predict, background)
sv = explainer.shap_values(X_scaled, nsamples=200)

fig, ax = plt.subplots(figsize=(js.SINGLE_COL, js.SINGLE_COL))
explanation = shap.Explanation(values=sv, data=X_scaled,
                                feature_names=["CTAB", "HAuCl₄", "AA", "GSH"])
shap.plots.beeswarm(explanation, show=False, max_display=4, color=SHAP_CMAP)
fig.set_size_inches(js.SINGLE_COL, js.SINGLE_COL)
js.save_fig(fig, "Fig3_SHAP_beeswarm")
print("  -> figures/Fig3_SHAP_beeswarm.pdf")
print("Done.")
