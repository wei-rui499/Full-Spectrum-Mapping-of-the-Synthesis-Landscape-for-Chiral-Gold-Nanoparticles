"""
Figure 2 — UMAP-based topological manifold analysis of g-factor spectra.

Outputs:
  figures/Fig2_UMAP_4cluster.pdf          — UMAP scatter with 4 clusters
  figures/Fig2_spectra_by_cluster.pdf     — mean spectra per cluster
"""
import numpy as np
import matplotlib.pyplot as plt
import journal_style as js
from pipeline import load_data, run_fpca, cluster_kmeans, umap_embed, RANDOM_STATE

js.apply_style()
colors = js.CLUSTER_COLORS_4

# ── Load & process ──────────────────────────────────────
wavelengths, spectra, params, sids = load_data()
fpc_scores = run_fpca(wavelengths, spectra)
labels, fpc_std = cluster_kmeans(fpc_scores)
embedding = umap_embed(fpc_std)

# ── Fig 2b: UMAP scatter (point size ∝ |g-factor|) ──────
g_abs = np.abs(params["g_factor"].values)
s_min, s_max = 40, 156
sizes = s_min + (g_abs - g_abs.min()) / (g_abs.max() - g_abs.min()) * (s_max - s_min)
rng = np.random.RandomState(RANDOM_STATE)
emb_jit = embedding + rng.normal(0, 0.12, size=embedding.shape)

fig, ax = plt.subplots(figsize=(js.SINGLE_COL, js.SINGLE_COL))
for k in range(4):
    mask = labels == k
    ax.scatter(emb_jit[mask, 0], emb_jit[mask, 1],
               c=colors[k], s=sizes[mask], alpha=0.85,
               edgecolors="white", linewidths=0.3,
               label=f"Cluster {k} (n={mask.sum()})", zorder=3)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
leg1 = ax.legend(loc="upper left", markerscale=0.8)
ax.add_artist(leg1)
# g-factor size legend
g_vals = [g_abs.min(), np.median(g_abs), g_abs.max()]
s_vals = [s_min, (s_min + s_max) / 2, s_max]
legend_handles = [plt.scatter([], [], s=sv, c="gray", alpha=0.6,
                   edgecolors="white", linewidths=0.3, label=f"|g|={gv:.3f}")
                   for gv, sv in zip(g_vals, s_vals)]
ax.legend(handles=legend_handles, loc="lower right", title="|g-factor|",
          frameon=False, labelspacing=1.2, handletextpad=0.5)
fig.tight_layout()
js.save_fig(fig, "Fig2_UMAP_4cluster")
print("  -> figures/Fig2_UMAP_4cluster.pdf")

# ── Fig 2e: Spectra by cluster ──────────────────────────
fig, ax = plt.subplots(figsize=(js.SINGLE_COL * 1.12, js.SINGLE_COL))
for k in range(4):
    mask = labels == k
    mu = spectra[mask].mean(axis=0)
    sd = spectra[mask].std(axis=0)
    ax.fill_between(wavelengths, mu - sd, mu + sd, color=colors[k], alpha=0.2)
    ax.plot(wavelengths, mu, color=colors[k], lw=1.2,
            label=f"Cluster {k} (n={mask.sum()})")
ax.axhline(0, color="grey", lw=0.5, ls="--", zorder=0)
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("g-factor")
ax.set_xlim(400, 1000); ax.legend(loc="best")
fig.tight_layout()
js.save_fig(fig, "Fig2_spectra_by_cluster")
print("  -> figures/Fig2_spectra_by_cluster.pdf")
print("Done.")
