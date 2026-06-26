"""
Core pipeline for chiral Au NP synthesis landscape mapping.

Shared functions imported by all figure scripts:
  load_data()       — load spectra + experiment parameters
  run_fpca()        — B-spline smoothing + PCA
  cluster_kmeans()  — KMeans clustering (k=4)
  umap_embed()      — UMAP 2D embedding
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import make_lsq_spline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters (fixed across all analyses) ──────────────────────
N_BASIS = 25
N_FPC = 3
N_CLUSTER = 4
UMAP_NEIGHBORS = 10
UMAP_MIN_DIST = 0.15
RANDOM_STATE = 42


def load_data():
    """Load g-factor spectra and experiment parameters.

    Returns
    -------
    wavelengths : (351,) array, nm
    spectra     : (80, 351) array, g-factor values
    params      : DataFrame with columns [CTAB, Au, AA, GSH, g_factor, Batch, Well]
    sample_ids  : list of str, "B{batch}-W{well}"
    """
    df_spec = pd.read_csv(DATA_DIR / "g_factor_spectra.csv", encoding="latin-1")
    df_spec.rename(columns={df_spec.columns[0]: "wavelength_nm"}, inplace=True)
    wavelengths = df_spec["wavelength_nm"].values.astype(float)
    spectral_cols = [c for c in df_spec.columns if c != "wavelength_nm"]

    df_exp = pd.read_csv(DATA_DIR / "experiment_history.csv", encoding="latin-1")
    df_exp["Batch"] = df_exp["Batch"].astype(int)
    df_exp["Well"] = df_exp["Well"].astype(int)

    def _parse_bw(col):
        for pat in [r"432-(\d+)-TESTAU-(\d)", r"432-YIBANNONGDU-(\d+)-(\d)",
                    r"432-(\d+)-(\d)\s*-\s*RawData", r"432-(\d+)-(\d)"]:
            m = re.search(pat, col)
            if m:
                return int(m.group(1)), int(m.group(2))
        return None, None

    df_map = pd.DataFrame([{"col_name": c, "Batch": _parse_bw(c)[0],
                             "Well": _parse_bw(c)[1]} for c in spectral_cols])
    for i, row in df_map.iterrows():
        if pd.isna(row["Batch"]):
            df_map.at[i, "Batch"] = df_exp.iloc[i]["Batch"]
            df_map.at[i, "Well"] = df_exp.iloc[i]["Well"]
    df_map["Batch"] = df_map["Batch"].astype(int)
    df_map["Well"] = df_map["Well"].astype(int)

    df_merged = df_map.merge(df_exp, on=["Batch", "Well"], how="inner").reset_index(drop=True)

    spectra = np.zeros((len(df_merged), len(wavelengths)))
    for idx, row in df_merged.iterrows():
        spectra[idx] = df_spec[row["col_name"]].values.astype(float)

    params = df_merged[["CTAB", "Au", "AA", "GSH", "g-factor", "Batch", "Well"]].copy()
    params.rename(columns={"g-factor": "g_factor"}, inplace=True)
    params = params.reset_index(drop=True)
    sample_ids = [f"B{int(r.Batch)}-W{int(r.Well)}" for _, r in params.iterrows()]
    return wavelengths, spectra, params, sample_ids


def run_fpca(wavelengths, spectra):
    """B-spline smoothing + PCA (functional PCA).

    Parameters
    ----------
    wavelengths : (351,) array
    spectra     : (80, 351) array

    Returns
    -------
    fpc_scores : (80, 3) array — first 3 fPC scores
    """
    n_samples, n_points = spectra.shape
    n_internal = N_BASIS - 4
    t_int = np.linspace(wavelengths.min(), wavelengths.max(), n_internal + 2)[1:-1]
    t = np.concatenate([[wavelengths.min()] * 4, t_int, [wavelengths.max()] * 4])

    smoothed = np.zeros_like(spectra)
    for i in range(n_samples):
        try:
            spl = make_lsq_spline(wavelengths, spectra[i], t, k=3)
            smoothed[i] = spl(wavelengths)
        except Exception:
            smoothed[i] = spectra[i]

    pca = PCA(n_components=N_FPC)
    fpc_scores = pca.fit_transform(smoothed)
    evr = pca.explained_variance_ratio_
    print(f"fPCA: {evr.sum()*100:.1f}% variance "
          f"({', '.join(f'{v*100:.1f}%' for v in evr)})")
    return fpc_scores


def cluster_kmeans(fpc_scores):
    """KMeans clustering on standardized fPC scores.

    Returns
    -------
    labels     : (80,) array, cluster labels 0-3
    fpc_std    : (80, 3) array, standardized fPC scores
    """
    scaler = StandardScaler()
    fpc_std = scaler.fit_transform(fpc_scores)
    km = KMeans(n_clusters=N_CLUSTER, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(fpc_std)
    for cl in range(N_CLUSTER):
        print(f"  C{cl}: n={np.sum(labels == cl)}")
    return labels, fpc_std


def umap_embed(fpc_std):
    """UMAP 2D embedding of standardized fPC scores.

    Returns
    -------
    embedding : (80, 2) array
    """
    import umap
    reducer = umap.UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                        metric="euclidean", random_state=RANDOM_STATE)
    embedding = reducer.fit_transform(fpc_std)
    # rotate 180° for visual consistency
    embedding = np.column_stack([-embedding[:, 0], -embedding[:, 1]])
    return embedding


def get_cluster_colors():
    """Return the 4-cluster color palette."""
    import style as js
    return js.CLUSTER_COLORS_4
