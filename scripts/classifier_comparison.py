"""
Multi-classifier benchmark and logistic regression diagnostics for cluster classification.

Outputs:
  figures/SI_CLF_model_comparison.pdf
  figures/SI_LR_confusion_matrix.pdf
  figures/SI_LR_ROC_curves.pdf
  figures/SI_LR_SHAP_beeswarm.pdf
"""
import runtime  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import shap
import style as js
from pipeline import load_data, run_fpca, cluster_kmeans, RANDOM_STATE

js.apply_style()
PARAMS = ["CTAB", "Au", "AA", "GSH"]
CLASS_NAMES = ["C1", "C2", "C3"]
CLASS_COLORS = [js.CLUSTER_COLORS_4[1], js.CLUSTER_COLORS_4[2], js.CLUSTER_COLORS_4[3]]
SHAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "shap_cmap", [js.PALETTE_30[2], "#FFFFFF", js.PALETTE_30[23]], N=256)
MODEL_TYPES = {'ExtraTrees': 'tree', 'AdaBoost': 'tree', 'SVC': 'kernel',
               'GaussianProcess': 'kernel', 'LDA': 'linear', 'LogisticReg': 'linear',
               'Stacking': 'ensemble'}
TYPE_COLORS = {'tree': js.PALETTE_30[10], 'kernel': js.PALETTE_30[15],
               'linear': js.PALETTE_30[2], 'ensemble': js.PALETTE_30[23]}

# ── Load & subset ───────────────────────────────────────
wavelengths, spectra, params_df, sids = load_data()
fpc_scores = run_fpca(wavelengths, spectra)
labels_all, _ = cluster_kmeans(fpc_scores)
X_all = params_df[PARAMS].values.astype(float)
mask = labels_all != 0
X = X_all[mask]; y_raw = labels_all[mask]; y = y_raw - 1
print(f"n={len(y)}: C1={np.sum(y==0)}, C2={np.sum(y==1)}, C3={np.sum(y==2)}")

# ════════════════════════════════════════════════════════
# SI Fig S13: Model comparison
# ════════════════════════════════════════════════════════
models = {
    'ExtraTrees': Pipeline([('scaler', StandardScaler()),
        ('model', ExtraTreesClassifier(n_estimators=500, max_depth=10,
            class_weight='balanced', random_state=RANDOM_STATE))]),
    'AdaBoost': Pipeline([('scaler', StandardScaler()),
        ('model', AdaBoostClassifier(n_estimators=300, learning_rate=0.5,
            random_state=RANDOM_STATE))]),
    'SVC': Pipeline([('scaler', StandardScaler()),
        ('model', SVC(C=10.0, kernel='rbf', gamma='scale',
            class_weight='balanced', probability=True, random_state=RANDOM_STATE))]),
    'GaussianProcess': Pipeline([('scaler', StandardScaler()),
        ('model', GaussianProcessClassifier(
            kernel=ConstantKernel(1.0)*Matern(length_scale=1.0, nu=1.5),
            n_restarts_optimizer=10, random_state=RANDOM_STATE))]),
    'LDA': Pipeline([('scaler', StandardScaler()),
        ('model', LinearDiscriminantAnalysis(solver='svd'))]),
    'LogisticReg': Pipeline([('scaler', StandardScaler()),
        ('model', LogisticRegression(C=55.0, penalty='l1', solver='saga',
            class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE))]),
}

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
results = {}
for name, pipe in models.items():
    from sklearn.model_selection import cross_validate as cv_fn
    cv_res = cv_fn(pipe, X, y, cv=rskf, scoring=['balanced_accuracy', 'f1_macro'])
    loo_pred = np.zeros(len(y), dtype=int)
    for tr, te in LeaveOneOut().split(X):
        p = clone(pipe); p.fit(X[tr], y[tr])
        loo_pred[te] = p.predict(X[te])
    results[name] = {
        'cv_bacc': cv_res['test_balanced_accuracy'],
        'cv_f1': cv_res['test_f1_macro'],
        'loo_bacc': balanced_accuracy_score(y, loo_pred),
        'loo_f1': f1_score(y, loo_pred, average='macro')
    }
    print(f"  {name:<20} LOO BAcc={results[name]['loo_bacc']:.4f}  F1={results[name]['loo_f1']:.4f}")

# Stacking
ranked = sorted(results.keys(), key=lambda k: results[k]['loo_bacc'], reverse=True)
top3 = ranked[:3]
stack = StackingClassifier(
    estimators=[(n, clone(models[n])) for n in top3],
    final_estimator=LogisticRegression(C=10.0, solver='saga', class_weight='balanced',
                                        max_iter=5000, random_state=RANDOM_STATE),
    cv=5, passthrough=True)
pipe_stack = Pipeline([('scaler', StandardScaler()), ('model', stack)])
cv_res_s = cv_fn(pipe_stack, X, y, cv=rskf, scoring=['balanced_accuracy', 'f1_macro'])
loo_pred_s = np.zeros(len(y), dtype=int)
for tr, te in LeaveOneOut().split(X):
    p = clone(pipe_stack); p.fit(X[tr], y[tr])
    loo_pred_s[te] = p.predict(X[te])
results['Stacking'] = {
    'cv_bacc': cv_res_s['test_balanced_accuracy'],
    'cv_f1': cv_res_s['test_f1_macro'],
    'loo_bacc': balanced_accuracy_score(y, loo_pred_s),
    'loo_f1': f1_score(y, loo_pred_s, average='macro')
}

# Plot comparison
names = list(results.keys()); n_m = len(names)
x = np.arange(n_m); bw = 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(js.DOUBLE_COL, js.SINGLE_COL * 0.85))
for ax, mkey, mlabel in [(ax1, 'bacc', 'Balanced Accuracy'), (ax2, 'f1', 'F1 Macro')]:
    cv_key = f'cv_{mkey}'; loo_key = f'loo_{mkey}'
    cv_means = [np.mean(results[n][cv_key]) for n in names]
    cv_stds = [np.std(results[n][cv_key]) for n in names]
    loo_vals = [results[n][loo_key] for n in names]
    tc = [TYPE_COLORS[MODEL_TYPES.get(n, 'ensemble')] for n in names]
    ax.bar(x - bw/2, cv_means, bw, yerr=cv_stds, capsize=2, color=tc, alpha=0.55,
           edgecolor='white', lw=0.5, label='CV')
    ax.bar(x + bw/2, loo_vals, bw, color=tc, alpha=0.95, edgecolor='white', lw=0.5,
           label='LOO')
    for xi, v in enumerate(loo_vals):
        ax.text(xi + bw/2, v + 0.012, f'{v:.3f}', ha='center', va='bottom',
                fontsize=js.FONTSIZE_TICK - 2, fontweight='bold', rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=js.FONTSIZE_TICK - 1)
    ax.set_ylabel(mlabel); ax.set_ylim(0.45, 1.05)
    if mlabel == 'Balanced Accuracy':
        ax.legend(fontsize=js.FONTSIZE_LEGEND, frameon=False, loc='lower right')
js.save_fig(fig, "SI_CLF_model_comparison")
print("  -> figures/SI_CLF_model_comparison.pdf")

# ════════════════════════════════════════════════════════
# SI Fig S14: LR detailed diagnostics
# ════════════════════════════════════════════════════════
lr_pipe = models['LogisticReg']
lr_pipe.fit(X, y)
lr_model = lr_pipe['model']
X_scaled = lr_pipe['scaler'].transform(X)

# LOO
loo_pred = np.zeros(len(y), dtype=int)
loo_proba = np.zeros((len(y), 3))
for tr, te in LeaveOneOut().split(X):
    p = clone(lr_pipe); p.fit(X[tr], y[tr])
    loo_pred[te] = p.predict(X[te])
    loo_proba[te] = p.predict_proba(X[te])
bacc = balanced_accuracy_score(y, loo_pred)
f1 = f1_score(y, loo_pred, average='macro')
cm = confusion_matrix(y, loo_pred)

# S14b: Confusion matrix
fig, ax = plt.subplots(figsize=(js.SINGLE_COL * 0.9, js.SINGLE_COL * 0.85))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})", ha="center", va="center",
                fontsize=js.FONTSIZE_TICK, color="white" if cm_norm[i,j] > 0.5 else "black")
ax.set_xticks([0,1,2]); ax.set_xticklabels(CLASS_NAMES)
ax.set_yticks([0,1,2]); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"LOO Confusion Matrix\nBAcc={bacc:.3f}  F1={f1:.3f}", fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.85, label="Recall")
js.save_fig(fig, "SI_LR_confusion_matrix")
print("  -> figures/SI_LR_confusion_matrix.pdf")

# S14c: SHAP beeswarm (per-class)
coef = lr_model.coef_
intercept = lr_model.intercept_
sv = X_scaled[np.newaxis, :, :] * coef[:, np.newaxis, :]
fig, axes = plt.subplots(1, 3, figsize=(js.DOUBLE_COL, js.SINGLE_COL * 1.15))
for ci, (ax, cname, col) in enumerate(zip(axes, CLASS_NAMES, CLASS_COLORS)):
    expl = shap.Explanation(values=sv[ci],
        base_values=np.full(len(X_scaled), float(intercept[ci])),
        data=X_scaled, feature_names=["CTAB", "HAuCl₄", "AA", "GSH"])
    plt.sca(ax)
    shap.plots.beeswarm(expl, show=False, max_display=4, color=SHAP_CMAP)
    fig.set_size_inches(js.DOUBLE_COL, js.SINGLE_COL * 1.15)
    ax.set_title(cname, color=col, fontweight="bold")
fig.suptitle("SHAP Beeswarm — Logistic Regression", fontweight="bold")
fig.subplots_adjust(left=0.12, right=0.97, wspace=0.5)
js.save_fig(fig, "SI_LR_SHAP_beeswarm")
print("  -> figures/SI_LR_SHAP_beeswarm.pdf")

# S14d: ROC curves
fig, ax = plt.subplots(figsize=(js.SINGLE_COL, js.SINGLE_COL))
y_bin = label_binarize(y, classes=[0,1,2])
for i, (cname, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], loo_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=1.5, label=f"{cname} (AUC={roc_auc:.3f})")
ax.plot([0,1], [0,1], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=js.FONTSIZE_LEGEND, frameon=False, loc="lower right")
js.save_fig(fig, "SI_LR_ROC_curves")
print("  -> figures/SI_LR_ROC_curves.pdf")

print(f"\nDone. LR LOO BAcc={bacc:.4f}, F1={f1:.4f}")
