# Full-Spectrum-Mapping-of-the-Synthesis-Landscape-for-Chiral-Gold-Nanoparticles

This repository contains the code for reproducing the analyses and figures in:

> Yang, Z.-B., Li, J.-T., Wang, X.-Y., Zhang, Z.-X., Wu, T., Fei, Q., Feng, G., Zhang, N.-N.\*, Kumacheva, E.\*, & Liu, K.\* *Full-Spectrum Mapping of the Synthesis Landscape for Chiral Gold Nanoparticles*. XXXX(2026).

## Abstract

Chiral plasmonic nanoparticles show strong chiroptical activity, but their synthesis remains difficult to rationalize because synthesis parameters, nanoparticle morphology and the asymmetry factor (g-factor) are coupled in a high-dimensional landscape. Here we establish a data-driven framework for chiral gold 432 helicoids by using the full g-factor spectrum, rather than a scalar peak value, as the coordinate of chemical space. Gryffin-driven autonomous synthesis identifies a high-|g-factor| formulation with |g-factor| = 0.24, while spectral manifold learning converts 80 outcomes into a topological landscape with four structural branches. The map reveals that gold precursor loading enables access to three high-|g-factor| branches, whereas L-glutathione loading directs branch selection. Symbolic regression identifies a d⁴L descriptor, linking gap depth (d) and particle length (L) to the |g-factor|. Low-signal regions, normally discarded during optimisation, contain previously unreported chiral rhombic dodecahedra, connecting achiral octahedra to final 432 helicoids and redirecting machine learning toward mechanistic synthesis understanding.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── g_factor_spectra.csv          # 80 g-factor spectra (351 wavelengths, 400–1100 nm)
│   ├── experiment_history.csv        # Synthesis parameters, g-factor, cluster labels
│   ├── sem_geometry.csv              # SEM-measured geometry (L, d, w, θ) for 15 particles
│   ├── fdtd_simulations.xlsx         # 50 FDTD simulated structures with |g-factor|
├── scripts/
│   ├── journal_style.py              # Shared plotting style (Nature-family formatting)
│   ├── pipeline.py                   # Core pipeline: data loading, fPCA, clustering, UMAP
│   ├── fig2_umap_landscape.py        # Figure 2: UMAP manifold & cluster spectra
│   ├── fig3_param_mapping.py         # Figure 3: GP regression, SHAP, correlation matrix
│   ├── fig4_classification_sisso.py  # Figure 4: Classification boundaries
│   ├── si_gp_regression.py           # SI Figures S11–S12: GP diagnostics
│   ├── si_classification.py          # SI Figures S13–S14: Classification comparison & LR SHAP
│   ├── si_fdtd_validation.py         # SI Figure S18: FDTD cross-scale validation
│   └── si_sisso_validation.py        # SI Figure S17: SISSO d⁴L robustness validation
```


## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/wei-rui499/Full-Spectrum-Mapping-of-the-Synthesis-Landscape-for-Chiral-Gold-Nanoparticles.git
cd Full-Spectrum-Mapping-of-the-Synthesis-Landscape-for-Chiral-Gold-Nanoparticles

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

All scripts are run from the repository root:

```bash
# Main figures
python scripts/fig2_umap_landscape.py        # Figure 2 (~30 s)
python scripts/fig3_param_mapping.py         # Figure 3 (~2 min)
python scripts/fig4_classification_sisso.py  # Figure 4 (~10 s)

# SI figures
python scripts/si_gp_regression.py           # SI S11–S12 (~3 min)
python scripts/si_classification.py          # SI S13–S14 (~5 min)
python scripts/si_sisso_validation.py        # SI S17 (~10 min)
python scripts/si_fdtd_validation.py         # SI S18 (~20 s)
```

Generated figures are saved to the `figures/` directory as PDF (600 DPI) and PNG preview files.

## Key Results

| Analysis | Method | Key Metric |
|----------|--------|------------|
| Spectral manifold learning | fPCA (25 B-spline basis → 3 fPC, 98.0% var) → KMeans k=4 → UMAP | Silhouette = 0.594 |
| GP regression | Matérn-5/2 kernel + WhiteKernel | LOO R² = 0.817, RMSE = 0.0298 |
| Classification (C1–C3) | Logistic Regression (C=55, L1) | LOO BAcc = 0.905, F1 = 0.881 |
| SISSO descriptor | d⁴L = d⁴ × L | LOO R² = 0.760, p < 0.004 (300 perm) |

## Data

- `g_factor_spectra.csv`: 80 samples × 351 wavelengths (400–1100 nm, 2 nm step), dimensionless g-factor
- `experiment_history.csv`: Synthesis parameters (CTAB, HAuCl₄, AA, GSH in mM), max |g-factor|, cluster labels
- `sem_geometry.csv`: SEM-measured geometric parameters (L, d, w, θ) for 15 representative particles
- `fdtd_simulations.xlsx`: 50 FDTD-simulated 432 helicoid structures with geometric parameters and |g-factor|

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yang2026fullspectrum,
  title={Full-Spectrum Mapping of the Synthesis Landscape for Chiral Gold Nanoparticles},
  author={Yang, Zhi-Bo and Li, Jia-Tong and Wang, Xue-Yao and Zhang, Zi-Xuan and Wu, Tianyi and Fei, Qiang and Feng, Guodong and Zhang, Ning-Ning and Kumacheva, Eugenia and Liu, Kun},
  journal={XXXXXX},
  year={2026},
  doi={10.1038/XXXXXXX}
}
```

## Contact

For questions about the code, please open an issue on this repository or contact the corresponding authors.
