"""
Figure 4 — Classification and SISSO geometric descriptor.

Outputs:
  figures/Fig4_classification_boundaries.pdf
  figures/Fig4_geometric_distributions.pdf
  figures/Fig4_PCA_biplot.pdf
  figures/Fig4_SISSO_d4L.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import journal_style as js
from pipeline import load_data, run_fpca, cluster_kmeans, RANDOM_STATE

js.apply_style()
PARAMS = ["CTAB", "Au", "AA", "GSH"]
CLASS_NAMES = ["C1", "C2", "C3"]
CLASS_COLORS = [js.CLUSTER_COLORS_4[1], js.CLUSTER_COLORS_4[2], js.CLUSTER_COLORS_4[3]]

# ── Load & cluster ──────────────────────────────────────
wavelengths, spectra, params_df, sids = load_data()
fpc_scores = run_fpca(wavelengths, spectra)
labels_all, _ = cluster_kmeans(fpc_scores)

X_all = params_df[PARAMS].values.astype(float)
mask = labels_all != 0
X = X_all[mask]; y_raw = labels_all[mask]; y = y_raw - 1
print(f"Classification: n={len(y)} (C1={np.sum(y==0)}, C2={np.sum(y==1)}, C3={np.sum(y==2)})")

# ════════════════════════════════════════════════════════
# Fig 4a: Decision boundaries (GSH vs CTAB/Au/AA)
# ════════════════════════════════════════════════════════
pipe = Pipeline([("scaler", StandardScaler()),
    ("model", LogisticRegression(C=55.0, penalty="l1", solver="saga",
                                  class_weight="balanced", max_iter=5000,
                                  random_state=RANDOM_STATE))])
pipe.fit(X, y)

other_params = [("CTAB", 0), ("Au", 1), ("AA", 2)]
GSH_IDX = 3
fig, axes = plt.subplots(1, 3, figsize=(js.DOUBLE_COL, js.SINGLE_COL * 0.58))
for ax, (other_name, other_idx) in zip(axes, other_params):
    X2 = X[:, [other_idx, GSH_IDX]]
    pipe2 = Pipeline([("scaler", StandardScaler()),
        ("model", LogisticRegression(C=55.0, penalty="l1", solver="saga",
                                      class_weight="balanced", max_iter=5000,
                                      random_state=RANDOM_STATE))])
    pipe2.fit(X2, y)
    x0_min, x0_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    x1_min, x1_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, 200),
                            np.linspace(x1_min, x1_max, 200))
    Z = pipe2.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)
    ax.contourf(xx0, xx1, Z, alpha=0.12, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=[CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2]])
    for cl in range(3):
        m_cl = y == cl
        ax.scatter(X2[m_cl, 0], X2[m_cl, 1], c=CLASS_COLORS[cl], s=40, alpha=0.85,
                   edgecolors="white", lw=0.4, label=CLASS_NAMES[cl], zorder=3)
    ax.set_xlabel(f"{other_name} (mM)" + (" / μM" if other_name == "GSH" else ""))
    ax.set_ylabel("GSH (μM)")
    if other_name == "CTAB":
        ax.legend(fontsize=js.FONTSIZE_LEGEND - 1, frameon=False, loc="upper right")
js.save_fig(fig, "Fig4_classification_boundaries")
print("  -> figures/Fig4_classification_boundaries.pdf")

print("Done.")
