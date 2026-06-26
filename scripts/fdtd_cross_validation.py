"""
FDTD cross-scale validation of the d⁴L empirical descriptor across simulation (n=50)
and experimental (n=15) datasets (3×2 panels).

Outputs:
  figures/SI_FDTD_validation.pdf
"""
import sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from matplotlib.patches import Patch
from style import (
    apply_style, save_fig, add_panel_label,
    CLUSTER_COLORS_4, DOUBLE_COL,
    FONTSIZE_TICK, FONTSIZE_LABEL, FONTSIZE_LEGEND,
    PALETTE_30,
)

apply_style()
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "figures"

# ── Colors ──────────────────────────────────────────
C_FDTD   = PALETTE_30[2]    # deep blue   FDTD scatter
C_FIT_F  = PALETTE_30[4]    # mid blue    FDTD fit line
C_FIT_E  = PALETTE_30[23]   # deep brick  experimental fit line
C_NEG    = PALETTE_30[25]   # brick red   d-theta anti-correlation
C_BAR_HI = PALETTE_30[25]   # brick red   highlighted bars
C_BAR_LO = PALETTE_30[7]    # light grey-blue  normal bars
CMAP_EXP = {1: CLUSTER_COLORS_4[1], 2: CLUSTER_COLORS_4[2], 3: PALETTE_30[15]}

# ════════════════════════════════════════════════════
# 1. Load FDTD data (50 LHS structures)
# ════════════════════════════════════════════════════
df_fdtd = pd.read_excel(DATA_DIR / "fdtd_simulations.xlsx")
df_fdtd["d4L"] = df_fdtd["d"] ** 4 * df_fdtd["L"]
df_fdtd["d3t"] = df_fdtd["d"] ** 3 * df_fdtd["theta"]
y_fdtd = df_fdtd["|g|"].values

# ════════════════════════════════════════════════════
# 2. Load experimental geometry (15 particles)
# ════════════════════════════════════════════════════
df_exp = pd.read_csv(DATA_DIR / "sem_geometry.csv")
df_exp["d4L"] = df_exp["d_nm"]**4 * df_exp["L_nm"]
df_exp["d3t"] = df_exp["d_nm"]**3 * df_exp["theta_deg"]
y_exp = df_exp["g_factor"].values
exp_colors = [CMAP_EXP.get(c, PALETTE_30[20]) for c in df_exp["cluster"]]

# ════════════════════════════════════════════════════
# 3. Compute statistics
# ════════════════════════════════════════════════════
loo = LeaveOneOut()

# — a: d4L linear fits —
sl_f, ic_f, _, _, _ = stats.linregress(df_fdtd["d4L"], y_fdtd)
sl_e, ic_e, _, _, _ = stats.linregress(df_exp["d4L"], y_exp)

X_f4 = df_fdtd["d4L"].values.reshape(-1, 1)
pred_f4, true_f4 = [], []
for tr, te in loo.split(X_f4):
    pred_f4.append(LinearRegression().fit(X_f4[tr], y_fdtd[tr]).predict(X_f4[te])[0])
    true_f4.append(y_fdtd[te][0])
loo_r2_fd4L = r2_score(true_f4, pred_f4)

X_e4 = df_exp["d4L"].values.reshape(-1, 1)
pred_e4, true_e4 = [], []
for tr, te in loo.split(X_e4):
    pred_e4.append(LinearRegression().fit(X_e4[tr], y_exp[tr]).predict(X_e4[te])[0])
    true_e4.append(y_exp[te][0])
loo_r2_ed4L = r2_score(true_e4, pred_e4)

# — b: FDTD d3θ optimal —
X_f3 = df_fdtd["d3t"].values.reshape(-1, 1)
pred_f3, true_f3 = [], []
for tr, te in loo.split(X_f3):
    pred_f3.append(LinearRegression().fit(X_f3[tr], y_fdtd[tr]).predict(X_f3[te])[0])
    true_f3.append(y_fdtd[te][0])
loo_r2_d3t = r2_score(true_f3, pred_f3)
sl_d3t, ic_d3t, r_d3t, _, _ = stats.linregress(df_fdtd["d3t"], y_fdtd)

# — c: log-log power-law —
valid = (y_fdtd > 0) & (df_fdtd["theta"].values > 0)
log_g = np.log(y_fdtd[valid])
log_d = np.log(df_fdtd["d"].values[valid])
log_L = np.log(df_fdtd["L"].values[valid])
log_t = np.log(df_fdtd["theta"].values[valid])
A = np.column_stack([log_d, log_L, log_t, np.ones(valid.sum())])
coef, _, _, _ = np.linalg.lstsq(A, log_g, rcond=None)
x_ld_range = np.linspace(log_d.min(), log_d.max(), 100)
y_ld_line = coef[0]*x_ld_range + coef[1]*log_L.mean() + coef[2]*log_t.mean() + coef[3]

# — d: Exp d vs θ anti-correlation —
r_dt, p_dt = pearsonr(df_exp["d_nm"], df_exp["theta_deg"])
sl_dt, ic_dt, _, _, _ = stats.linregress(df_exp["d_nm"], df_exp["theta_deg"])
x_dt_range = np.linspace(df_exp["d_nm"].min()-2, df_exp["d_nm"].max()+2, 100)
n_dt = len(df_exp)
se_dt = np.sqrt(np.sum((df_exp["theta_deg"].values-(sl_dt*df_exp["d_nm"].values+ic_dt))**2)/(n_dt-2))
xm = df_exp["d_nm"].values.mean()
ss = np.sum((df_exp["d_nm"].values - xm)**2)
ci_band = stats.t.ppf(0.975, n_dt-2) * se_dt * np.sqrt(1/n_dt + (x_dt_range-xm)**2/ss)

# — e: d4L vs d3θ equivalence —
r_eq_f, _ = pearsonr(df_fdtd["d4L"], df_fdtd["d3t"])
r_eq_e, _ = pearsonr(df_exp["d4L"], df_exp["d3t"])

def norm01(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

# — f: Top-10 candidate descriptors —
d = df_fdtd["d"].values; L = df_fdtd["L"].values
w = df_fdtd["w"].values; th = df_fdtd["theta"].values
candidates = {
    r"$d^4 L$":       d**4 * L,
    r"$d^4$":         d**4,
    r"$d^3\theta$":   d**3 * th,
    r"$d^3 L$":       d**3 * L,
    r"$d^3$":         d**3,
    r"$d^5$":         d**5,
    r"$d^4/L$":       d**4 / L,
    r"$d^2 L^2$":     d**2 * L**2,
    r"$d^2 L$":       d**2 * L,
    r"$d \cdot L$":   d * L,
}
cand_r = {k: abs(pearsonr(v, y_fdtd)[0]) for k, v in candidates.items()}
cand_sorted = sorted(cand_r.items(), key=lambda x: -x[1])
bar_labels = [k for k, _ in cand_sorted]
bar_vals   = [v for _, v in cand_sorted]
bar_colors = [C_BAR_HI if lbl in (r"$d^4 L$", r"$d^3\theta$") else C_BAR_LO
              for lbl, _ in cand_sorted]

# ════════════════════════════════════════════════════
# 4. Plot — 3×2
# ════════════════════════════════════════════════════
FIG_W = DOUBLE_COL
FIG_H = DOUBLE_COL * 1.18
MS = 22; LW = 1.3; ALF = 0.60

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.30, wspace=0.32,
                       left=0.09, right=0.97, top=0.97, bottom=0.06)

# ─── a: Cross-scale functional form consistency ────
ax = fig.add_subplot(gs[0, 0])
ax.scatter(df_fdtd["d4L"], y_fdtd, s=MS, color=C_FDTD, alpha=ALF, linewidths=0, zorder=2)
x_f = np.linspace(df_fdtd["d4L"].min(), df_fdtd["d4L"].max(), 200)
ax.plot(x_f, sl_f*x_f + ic_f, color=C_FIT_F, lw=LW+0.3, zorder=3,
        label=f"FDTD: slope = {sl_f:.2e}")
ax.scatter(df_exp["d4L"], y_exp, s=MS*3, c=exp_colors, edgecolors="white",
           linewidths=0.5, zorder=5)
x_e = np.linspace(df_exp["d4L"].min(), df_exp["d4L"].max(), 200)
ax.plot(x_e, sl_e*x_e + ic_e, color=C_FIT_E, lw=LW+0.3, ls="--", zorder=4,
        label=f"Exp: slope = {sl_e:.2e}")
ax.legend(fontsize=FONTSIZE_LEGEND, loc="lower right", handlelength=1.2,
          labelspacing=0.25, frameon=True, framealpha=0.85, edgecolor="none")
ax.set_xlabel(r"$d^4L$ (nm$^5$)", fontsize=FONTSIZE_LABEL)
ax.set_ylabel(r"|$g$-factor|", fontsize=FONTSIZE_LABEL)
ax.text(0.05, 0.93, f"Slope ratio = {sl_f/sl_e:.1f}x\nLinear $d^4L$ form",
        transform=ax.transAxes, fontsize=FONTSIZE_TICK, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
ax.text(0.98, 0.55, "FDTD\n(n=50)", transform=ax.transAxes,
        fontsize=FONTSIZE_TICK, color=C_FDTD, ha="right", va="center")
ax.text(0.98, 0.35, "Exp.\n(n=15)", transform=ax.transAxes,
        fontsize=FONTSIZE_TICK, color=C_FIT_E, ha="right", va="center")
add_panel_label(ax, "a")

# ─── b: FDTD d3θ optimal descriptor ─────────────────
ax = fig.add_subplot(gs[0, 1])
ax.scatter(df_fdtd["d3t"], y_fdtd, s=MS, color=C_FDTD, alpha=ALF, linewidths=0, zorder=2)
x_b = np.linspace(df_fdtd["d3t"].min(), df_fdtd["d3t"].max(), 200)
ax.plot(x_b, sl_d3t*x_b + ic_d3t, color=C_FIT_F, lw=LW+0.3, zorder=3)
ax.set_xlabel(r"$d^3\theta$ (nm$^3$·deg)", fontsize=FONTSIZE_LABEL)
ax.set_ylabel(r"|$g$-factor|", fontsize=FONTSIZE_LABEL)
ax.text(0.05, 0.93, f"FDTD optimal descriptor\nLOO R$^2$ = {loo_r2_d3t:.3f},  r = {r_d3t:.3f}",
        transform=ax.transAxes, fontsize=FONTSIZE_TICK, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
add_panel_label(ax, "b")

# ─── c: log-log power-law ───────────────────────────
ax = fig.add_subplot(gs[1, 0])
sc = ax.scatter(log_d, log_g, s=MS, c=log_t, cmap="RdYlBu_r", alpha=0.75,
                linewidths=0, zorder=3, vmin=log_t.min(), vmax=log_t.max())
cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.035, aspect=20)
cb.set_label(r"log($\theta$)", fontsize=FONTSIZE_TICK)
cb.ax.tick_params(labelsize=FONTSIZE_TICK-1, width=0.5, length=2)
cb.outline.set_linewidth(0.5)
ax.plot(x_ld_range, y_ld_line, color=C_FIT_F, lw=LW+0.3, zorder=4)
ax.set_xlabel(r"log($d$ / nm)", fontsize=FONTSIZE_LABEL)
ax.set_ylabel(r"log(|$g$-factor|)", fontsize=FONTSIZE_LABEL)
ax.text(0.05, 0.93, f"d exponent = {coef[0]:.2f}\n(experiment assumes d$^4$)",
        transform=ax.transAxes, fontsize=FONTSIZE_TICK, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
add_panel_label(ax, "c")

# ─── d: Experimental d vs θ anti-correlation ────────
ax = fig.add_subplot(gs[1, 1])
for cid, lbl in [(1, "C1"), (2, "C2"), (3, "C3")]:
    sub = df_exp[df_exp["cluster"] == cid]
    ax.scatter(sub["d_nm"], sub["theta_deg"], s=MS*2.8, color=CMAP_EXP[cid],
               label=lbl, edgecolors="white", linewidths=0.4, zorder=3)
ax.plot(x_dt_range, sl_dt*x_dt_range + ic_dt, color=C_NEG, lw=LW+0.3, ls="--", zorder=2)
ax.fill_between(x_dt_range, sl_dt*x_dt_range + ic_dt - ci_band,
                sl_dt*x_dt_range + ic_dt + ci_band, color=C_NEG, alpha=0.18, zorder=1)
ax.set_xlabel(r"Gap depth $d$ (nm)", fontsize=FONTSIZE_LABEL)
ax.set_ylabel(r"Gap angle $\theta$ (deg)", fontsize=FONTSIZE_LABEL)
ax.legend(fontsize=FONTSIZE_LEGEND, loc="upper right", title="Cluster",
          title_fontsize=FONTSIZE_LEGEND, frameon=True, framealpha=0.85, edgecolor="none")
p_str = f"= {p_dt:.3f}" if p_dt >= 0.001 else "< 0.001"
ax.text(0.05, 0.12, f"r = {r_dt:.3f},  p {p_str}\nLarge d => small theta",
        transform=ax.transAxes, fontsize=FONTSIZE_TICK, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
add_panel_label(ax, "d")

# ─── e: Equivalence d4L vs d3θ ──────────────────────
ax = fig.add_subplot(gs[2, 0])
xn_f = norm01(df_fdtd["d4L"].values); yn_f = norm01(df_fdtd["d3t"].values)
xn_e = norm01(df_exp["d4L"].values); yn_e = norm01(df_exp["d3t"].values)
ax.scatter(xn_f, yn_f, s=MS, color=C_FDTD, alpha=0.45, linewidths=0, zorder=2,
           label=f"FDTD  r={r_eq_f:.3f}")
ax.scatter(xn_e, yn_e, s=MS*3, c=exp_colors, edgecolors="white", linewidths=0.4,
           zorder=4, label=f"Exp.  r={r_eq_e:.3f}")
ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", zorder=1)
ax.set_xlabel(r"$d^4L$ (normalized)", fontsize=FONTSIZE_LABEL)
ax.set_ylabel(r"$d^3\theta$ (normalized)", fontsize=FONTSIZE_LABEL)
ax.set_xlim(-0.05, 1.10); ax.set_ylim(-0.05, 1.10)
ax.legend(fontsize=FONTSIZE_LEGEND, loc="upper left", frameon=True,
          framealpha=0.85, edgecolor="none")
ax.text(0.50, 0.07, "Equivalent descriptors:\nboth capture the same physical signal",
        transform=ax.transAxes, fontsize=FONTSIZE_TICK, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
add_panel_label(ax, "e")

# ─── f: Descriptor degeneracy bar chart ─────────────
ax = fig.add_subplot(gs[2, 1])
y_pos = np.arange(len(bar_labels))
ax.barh(y_pos, bar_vals, color=bar_colors, edgecolor="none", height=0.62)
ax.set_yticks(y_pos); ax.set_yticklabels(bar_labels, fontsize=FONTSIZE_TICK+0.5)
ax.set_xlabel(r"Pearson |$r$| vs |$g$-factor| (FDTD)", fontsize=FONTSIZE_LABEL)
ax.set_xlim(0, 1.12); ax.invert_yaxis()
for i, v in enumerate(bar_vals):
    ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=FONTSIZE_TICK-0.5, color="black")
ax.legend(handles=[Patch(color=C_BAR_HI, label=r"$d^4L$ and $d^3\theta$"),
                   Patch(color=C_BAR_LO, label="Other")],
          fontsize=FONTSIZE_LEGEND, loc="lower right", frameon=True,
          framealpha=0.85, edgecolor="none")
ax.text(0.05, 0.07, "Descriptor degeneracy:\nmultiple equivalent forms",
        transform=ax.transAxes, fontsize=FONTSIZE_TICK, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
add_panel_label(ax, "f")

# ════════════════════════════════════════════════════
# 5. Save & report
# ════════════════════════════════════════════════════
save_fig(fig, "SI_FDTD_validation", output_dir=OUTPUT_DIR)

print("\n=== Key Statistics ===")
print(f"a  FDTD d4L slope={sl_f:.3e}, LOO R2={loo_r2_fd4L:.4f}")
print(f"   Exp  d4L slope={sl_e:.3e}, LOO R2={loo_r2_ed4L:.4f}")
print(f"   Slope ratio = {sl_f/sl_e:.2f}x")
print(f"b  FDTD d3t LOO R2={loo_r2_d3t:.4f}, r={r_d3t:.4f}")
print(f"c  log-log d exponent = {coef[0]:.2f} (L={coef[1]:.2f}, theta={coef[2]:.2f})")
print(f"d  Exp d vs theta: r={r_dt:.4f}, p={p_dt:.4f}")
print(f"e  FDTD r={r_eq_f:.4f}, Exp r={r_eq_e:.4f}")
print(f"f  Top-3: {cand_sorted[:3]}")
