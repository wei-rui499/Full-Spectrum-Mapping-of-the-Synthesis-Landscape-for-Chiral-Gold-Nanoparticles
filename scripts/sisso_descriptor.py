"""
SISSO symbolic regression: d⁴L descriptor discovery and robustness validation (2×3 panels).

Outputs:
  figures/SI_SISSO_validation.pdf
"""
import sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import style as js
js.apply_style()

OUTPUT_DIR = ROOT / "figures"
SEED = 42
N_PERM = 300
N_BOOT = 1000

# ════════════════════════════════════════════════════
# 1. Load structural data
# ════════════════════════════════════════════════════
df = pd.read_csv(ROOT / "data" / "experiment_history.csv", encoding="latin-1")
mask = df["L_nm"].notna()
df_sub = df.loc[mask].copy()
features = ["L_nm", "d_nm", "w_nm", "theta_deg"]
feature_labels = ["L", "d", "w", "theta"]  # short names for descriptor formulas
X = df_sub[features].values.astype(float)
y = df_sub["g-factor"].values.astype(float)
n_samples = len(y)
print(f"n={n_samples}, features={feature_labels}")
print(f"g-factor: [{y.min():.4f}, {y.max():.4f}]")

# ════════════════════════════════════════════════════
# 2. Build feature space
# ════════════════════════════════════════════════════
def build_fs(Xf, fnames):
    pool = {n: Xf[:, i] for i, n in enumerate(fnames)}
    def _v(a): return np.all(np.isfinite(a)) and np.std(a) > 1e-12
    def _a(d, n, a):
        if n not in d and _v(a): d[n] = a
    r1 = dict(pool)
    for n, a in pool.items():
        _a(r1, f"{n}²", a**2); _a(r1, f"{n}³", a**3)
        _a(r1, f"√{n}", np.sqrt(np.abs(a)))
        if np.all(a != 0): _a(r1, f"1/{n}", 1.0/a)
    primary = list(fnames)
    for i in range(len(primary)):
        for j in range(len(primary)):
            if i == j: continue
            ni, nj = primary[i], primary[j]; ai, aj = pool[ni], pool[nj]
            _a(r1, f"{ni}+{nj}", ai+aj); _a(r1, f"{ni}-{nj}", ai-aj)
            _a(r1, f"{ni}·{nj}", ai*aj)
            if np.all(aj != 0): _a(r1, f"{ni}/{nj}", ai/aj)
    r2 = dict(r1)
    r1n, r1a = list(r1.keys()), [r1[n] for n in r1]
    for i in range(min(len(r1n), 50)):
        for j in range(i+1, len(r1n)):
            ni, nj = r1n[i], r1n[j]; ai, aj = r1a[i], r1a[j]
            _a(r2, f"({ni}+{nj})", ai+aj); _a(r2, f"({ni}-{nj})", ai-aj)
            _a(r2, f"({ni}·{nj})", ai*aj)
            if np.all(np.abs(aj) > 1e-15):
                with np.errstate(divide='ignore', invalid='ignore'): r = ai/aj
                if _v(r): _a(r2, f"({ni}/{nj})", r)
    sn = sorted(r2.keys(), key=lambda n: (len(n), n))
    P = np.column_stack([r2[n] for n in sn])
    _, uid = np.unique(np.round(P, 10), axis=1, return_index=True)
    uid = np.sort(uid); P = P[:, uid]; sn = [sn[i] for i in uid]
    return P, sn

print("Building feature space...")
Phi_all, names_all = build_fs(X, feature_labels)
print(f"  {Phi_all.shape[1]} unique features")

# ════════════════════════════════════════════════════
# 3. 1D SISSO — identify best descriptor
# ════════════════════════════════════════════════════
def so_1d(Phi, y_t):
    corrs = np.array([abs(pearsonr(Phi[:, j], y_t)[0]) for j in range(Phi.shape[1])])
    top200 = np.argsort(corrs)[::-1][:200]
    Phis = Phi[:, top200]; nss = [names_all[i] for i in top200]
    best_r2, best_j, best_pred = -np.inf, None, None
    for j in range(Phis.shape[1]):
        X1 = Phis[:, j:j+1]; yp = np.zeros(len(y_t))
        for tr, te in LeaveOneOut().split(X1):
            lr = LinearRegression().fit(X1[tr], y_t[tr]); yp[te] = lr.predict(X1[te])
        r2 = r2_score(y_t, yp)
        if r2 > best_r2: best_r2, best_j, best_pred = r2, j, yp.copy()
    return best_j, nss[best_j], best_r2, best_pred, Phis[:, best_j]

descr_idx, descr_name, loo_r2, loo_pred, d4L_vals = so_1d(Phi_all, y)
lr_full = LinearRegression().fit(d4L_vals.reshape(-1, 1), y)
full_r2 = r2_score(y, lr_full.predict(d4L_vals.reshape(-1, 1)))
print(f"\nBest 1D: {descr_name}, LOO R²={loo_r2:.4f}, Full R²={full_r2:.4f}")
print(f"  |g| = {lr_full.coef_[0]:.6g} · d⁴L + {lr_full.intercept_:.6g}")

# ════════════════════════════════════════════════════
# 4. Validation computations
# ════════════════════════════════════════════════════
rng = np.random.RandomState(SEED)

# (a) Y-randomization
print(f"\n(a) Y-randomization ({N_PERM}×)...")
perm_r2 = np.zeros(N_PERM)
for pi in range(N_PERM):
    _, _, pr2, _, _ = so_1d(Phi_all, rng.permutation(y))
    perm_r2[pi] = pr2
    if (pi+1) % 100 == 0: print(f"  {pi+1}/{N_PERM}")
p_perm = np.mean(perm_r2 >= loo_r2)
print(f"  obs={loo_r2:.4f}, null μ={perm_r2.mean():.4f}, p={p_perm:.4f}")

# (b) Leave-k-out
print("\n(b) Leave-k-out...")
lk_r2 = {}
for k in [1, 3, 5]:
    scores = []
    if k == 1:
        folds = list(LeaveOneOut().split(X))
    elif k == 3:
        all_f = list(LeavePOut(3).split(X))
        folds = [all_f[i] for i in rng.choice(len(all_f), size=min(100, len(all_f)), replace=False)]
    else:
        folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
    for tr, te in folds:
        try:
            _, _, r2t, _, _ = so_1d(Phi_all[tr], y[tr])
            scores.append(r2t)
        except: pass
    lk_r2[k] = np.array(scores)
    print(f"  k={k}: R²={lk_r2[k].mean():.3f}±{lk_r2[k].std():.3f}")

# (c) Bootstrap
print(f"\n(c) Bootstrap ({N_BOOT}×)...")
boot_s, boot_i = [], []
for _ in range(N_BOOT):
    idx = rng.choice(n_samples, size=n_samples, replace=True)
    lr_b = LinearRegression().fit(d4L_vals[idx].reshape(-1, 1), y[idx])
    boot_s.append(lr_b.coef_[0]); boot_i.append(lr_b.intercept_)
boot_s = np.array(boot_s); boot_i = np.array(boot_i)
print(f"  Slope 95% CI: [{np.percentile(boot_s,2.5):.4g}, {np.percentile(boot_s,97.5):.4g}]")
print(f"  Intercept 95% CI: [{np.percentile(boot_i,2.5):.4g}, {np.percentile(boot_i,97.5):.4g}]")

# (d) Top-20
top20_loo = np.zeros(20)
corrs_all = np.array([abs(pearsonr(Phi_all[:,j], y)[0]) for j in range(Phi_all.shape[1])])
top20idx = np.argsort(corrs_all)[::-1][:20]
for j, idx in enumerate(top20idx):
    X1 = Phi_all[:, idx:idx+1]; yp = np.zeros(n_samples)
    for tr, te in LeaveOneOut().split(X1):
        lr = LinearRegression().fit(X1[tr], y[tr]); yp[te] = lr.predict(X1[te])
    top20_loo[j] = r2_score(y, yp)

# (e) GP vs SISSO
kern = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5)+WhiteKernel(noise_level=0.01)
gp = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=10, alpha=1e-6,
                               normalize_y=True, random_state=SEED)
gp_pipe = Pipeline([("s", StandardScaler()), ("m", gp)])
gp_pred = np.zeros(n_samples)
for tr, te in LeaveOneOut().split(X):
    p = clone(gp_pipe); p.fit(X[tr], y[tr]); gp_pred[te] = p.predict(X[te])
gp_r2 = r2_score(y, gp_pred)
r_resid = pearsonr(y - gp_pred, y - loo_pred)[0]
print(f"  GP LOO R²={gp_r2:.4f}, SISSO LOO R²={loo_r2:.4f}, resid r={r_resid:.3f}")

# ════════════════════════════════════════════════════
# 5. Plot — 2×3
# ════════════════════════════════════════════════════
CLR = {1: js.CLUSTER_COLORS_4[1], 2: js.CLUSTER_COLORS_4[2], 3: js.CLUSTER_COLORS_4[3]}
df_sub["cluster_id"] = df_sub["Cluster"].astype(str).str[0].astype(int)
clusters = df_sub["cluster_id"].values

def _sv(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", format="pdf", dpi=js.DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf")

fig = plt.figure(figsize=(js.DOUBLE_COL * 1.08, js.DOUBLE_COL * 1.22))
gs = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.32,
                       left=0.08, right=0.97, top=0.95, bottom=0.06)

# (a) Y-randomization
ax = fig.add_subplot(gs[0, 0])
ax.hist(perm_r2, bins=30, color=js.PALETTE_30[2], alpha=0.7, edgecolor="white", lw=0.3)
ax.axvline(loo_r2, color=js.PALETTE_30[23], lw=2.0)
p_str = "p < 0.002" if p_perm < 0.002 else f"p = {p_perm:.3f}"
ax.text(0.97, 0.96, f"{N_PERM} permutations\nobs R²={loo_r2:.3f}\nnull μ={perm_r2.mean():.3f}\n{p_str}",
        transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9, lw=0.5))
ax.set_xlabel("LOO R² (Y-shuffled)"); ax.set_ylabel("Count")
js.add_panel_label(ax, "a")

# (b) Leave-k-out
ax = fig.add_subplot(gs[0, 1])
bp = ax.boxplot([lk_r2[1], lk_r2[3], lk_r2[5]], positions=[1,2,3], widths=0.35,
                patch_artist=True, showfliers=True,
                flierprops=dict(markersize=3, alpha=0.4, markerfacecolor="grey"),
                medianprops=dict(color="white", lw=1.2))
for p, k in zip(bp["boxes"], [1,3,5]):
    p.set_facecolor(js.PALETTE_30[23] if k==1 else js.PALETTE_30[2]); p.set_alpha(0.8)
ax.axhline(loo_r2, color=js.PALETTE_30[23], ls="--", lw=0.8, alpha=0.6,
           label=f"LOO R²={loo_r2:.3f}")
for i, k in enumerate([1,3,5]):
    ax.text(i+1, lk_r2[k].mean()+0.015, f"{lk_r2[k].mean():.3f}", ha="center",
            va="bottom", fontsize=7, fontweight="bold")
ax.set_xticks([1,2,3]); ax.set_xticklabels(["LOO\n(k=1)", "L3O\n(k=3)", "L5O\n(k=5)"])
ax.set_ylabel("Best 1D desc. R²"); ax.legend(fontsize=6.5, loc="lower left", framealpha=0.85)
js.add_panel_label(ax, "b")

# (c) Bootstrap slope
ax = fig.add_subplot(gs[1, 0])
ax.hist(boot_s * 1e11, bins=40, color=js.PALETTE_30[2], alpha=0.7, edgecolor="white", lw=0.3)
ci_lo = np.percentile(boot_s, 2.5) * 1e11
ci_hi = np.percentile(boot_s, 97.5) * 1e11
ax.axvline(boot_s.mean() * 1e11, color=js.PALETTE_30[23], lw=1.5)
ax.axvline(ci_lo, color="grey", ls="--", lw=0.8)
ax.axvline(ci_hi, color="grey", ls="--", lw=0.8)
ax.text(0.97, 0.96,
        f"Slope 95% CI\n{boot_s.mean()*1e11:.2f} [{ci_lo:.2f}, {ci_hi:.2f}] ×10⁻¹¹",
        transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9, lw=0.5))
ax.set_xlabel("Slope (×10⁻¹¹ nm⁻⁵)"); ax.set_ylabel("Count")
js.add_panel_label(ax, "c")

# (d) Bootstrap intercept
ax = fig.add_subplot(gs[1, 1])
ax.hist(boot_i, bins=40, color=js.PALETTE_30[15], alpha=0.7, edgecolor="white", lw=0.3)
ilo, ihi = np.percentile(boot_i, 2.5), np.percentile(boot_i, 97.5)
ax.axvline(boot_i.mean(), color=js.PALETTE_30[23], lw=1.5)
ax.axvline(ilo, color="grey", ls="--", lw=0.8); ax.axvline(ihi, color="grey", ls="--", lw=0.8)
ax.text(0.97, 0.96, f"Intercept 95% CI\n{boot_i.mean():.4f} [{ilo:.4f}, {ihi:.4f}]",
        transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9, lw=0.5))
ax.set_xlabel("Intercept"); ax.set_ylabel("Count")
js.add_panel_label(ax, "d")

# (e) Top-20 descriptors
ax = fig.add_subplot(gs[2, 0])
n_show = 20
top20_names = [names_all[i] for i in top20idx]
order = np.argsort(top20_loo)[::-1]
colors_d = [js.PALETTE_30[23] if top20_names[o] == descr_name else js.PALETTE_30[2] for o in order]
ax.barh(range(n_show), top20_loo[order], color=colors_d, edgecolor="white", lw=0.3, height=0.6)
for i, o in enumerate(order):
    ax.text(top20_loo[o] + 0.006, i, f"{top20_loo[o]:.3f}", va="center", fontsize=5.5, color="#222222")
ax.set_yticks(range(n_show))
ax.set_yticklabels([top20_names[o] for o in order], fontsize=4.5, family="monospace")
ax.set_xlabel("LOO R² (1D)"); ax.invert_yaxis()
js.add_panel_label(ax, "e")

# (f) GP vs SISSO residuals
ax = fig.add_subplot(gs[2, 1])
# Recompute GP LOO predictions to ensure correct shape
kern2 = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5)+WhiteKernel(noise_level=0.01)
gp2 = GaussianProcessRegressor(kernel=kern2, n_restarts_optimizer=10, alpha=1e-6,
                                normalize_y=True, random_state=SEED)
gp2_pipe = Pipeline([("s", StandardScaler()), ("m", gp2)])
gp2_pred = np.zeros(n_samples)
for tr, te in LeaveOneOut().split(X):
    p = clone(gp2_pipe); p.fit(X[tr], y[tr]); gp2_pred[te] = p.predict(X[te])
gp2_r2 = r2_score(y, gp2_pred)
r2_resid = pearsonr(y - gp2_pred, y - loo_pred)[0]

for cid in sorted(np.unique(clusters)):
    m = clusters == cid
    ax.scatter(y[m] - gp2_pred[m], y[m] - loo_pred[m], c=CLR.get(cid, "grey"), s=28,
               alpha=0.85, edgecolors="white", lw=0.3,
               label={1:"C1",2:"C2",3:"C3"}.get(cid, "other"))
lo = min((y - gp2_pred).min(), (y - loo_pred).min()) * 1.3
hi = max((y - gp2_pred).max(), (y - loo_pred).max()) * 1.3
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.axhline(0, color="grey", ls="-", lw=0.5, alpha=0.3)
ax.axvline(0, color="grey", ls="-", lw=0.5, alpha=0.3)
ax.set_xlabel("GP residual"); ax.set_ylabel("SISSO residual")
ax.legend(fontsize=5.5, loc="lower right", framealpha=0.85)
ax.text(0.04, 0.94,
        f"GP LOO R²={gp2_r2:.3f}\nSISSO LOO R²={loo_r2:.3f}\nresid r={r2_resid:.3f}",
        transform=ax.transAxes, fontsize=6, va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9, lw=0.5))
js.add_panel_label(ax, "f")

_sv(fig, "SI_SISSO_validation")
print("\nDone.")
