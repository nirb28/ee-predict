# ee-predict: Catalyst Enantiomeric Excess Prediction & Optimization

A machine-learning exploration project for predicting and optimizing catalyst enantiomeric excess (ee) using molecular descriptors and coordinate-descent optimization.

## Project Goal

Build regression models that predict the thermodynamic selectivity of a catalyst (reported as `ddG`, which is proportional to `% ee`) from computed molecular descriptors, then use those models to propose catalyst variants with improved selectivity.

## Repository Structure

```text
.
├── data/                          # Datasets used by notebooks
│   ├── Data.csv                   # Main 3D embedding dataset (1,850 rows)
│   ├── reduced_dim_space_ddG.csv  # Reduced 3D descriptor space + ddG (% ee)
│   ├── large_cat_desc_col_names.csv  # Full descriptor matrix (~3,973 features, 1,903 rows)
│   ├── merged_large_catalyst.csv  # Merged full-feature catalyst dataset
│   └── original-datasets/         # Raw upstream inputs
│       ├── Data.csv
│       └── cat_desc.csv
├── models/                        # Serialized models and helpers
│   ├── pls.joblib                 # PLS regression model on 3D reduced space
│   ├── pls_large.joblib           # PLS regression model on full descriptor set
│   └── high_corr_cols.txt         # List of highly-correlated descriptor columns
├── archive/                       # Older experimental scripts
│   └── CoordinateDescent.py       # Early coordinate-descent scratchpad
├── large/
│   └── eda.ipynb                  # Exploratory data analysis on full descriptor set
├── linear-regression.ipynb        # Baseline OLS & PLS on reduced 3D space
├── knn.ipynb                      # k-nearest neighbors exploration
├── cd_v2.ipynb / cd_v2_old.ipynb  # Coordinate-descent catalyst optimizer v2
├── cd_iterative.ipynb             # Iterative coordinate-descent optimization loop
├── CD_FindCatalyst.ipynb          # End-to-end search for high-ee catalysts
├── 052024_model_suite.ipynb       # Model benchmarking suite
├── 052124_feature_selection.ipynb # Feature-selection experiments
├── 052024_fa_copy.ipynb           # Factor-analysis experiments
├── Mordred_Descriptors.ipynb      # Descriptor generation with Mordred/RDKit
├── requirements.txt               # Pip freeze of the working environment
└── environment.yml                # Conda environment export (`ml2`)
```

## Data

| File | Rows (approx) | Columns | Description |
|------|--------------|---------|-------------|
| `data/Data.csv` | 1,850 | 5 | Catalyst ID + 3D embedding (`x`, `y`, `z`) + `ddG (% ee)` |
| `data/reduced_dim_space_ddG.csv` | 1,850 | 5 | Same as `Data.csv`; used as the clean reduced-space input |
| `data/large_cat_desc_col_names.csv` | 1,903 | ~3,975 | Full Mordred/RDKit descriptor matrix + `ddG` |
| `data/merged_large_catalyst.csv` | 1,850 | ~3,975 | Merged/processed version of the full descriptor set |

Rows with `ddG == 0` are typically filtered out before modeling, leaving ~318 non-zero samples in the reduced-space datasets.

## Methodology

1. **Descriptor generation** (`Mordred_Descriptors.ipynb`)  
   Compute molecular descriptors with `mordred` and `rdkit`.

2. **Dimensionality reduction** (`data/reduced_dim_space_ddG.csv`)  
   Reduce the full descriptor matrix to a 3D embedding (`x`, `y`, `z`) for visualization and baseline modeling.

3. **Baseline modeling** (`linear-regression.ipynb`, `052024_model_suite.ipynb`)  
   Train ordinary least squares (OLS), partial least squares (PLS), and factor-analysis models to predict `ddG (% ee)`.

4. **Iterative optimization** (`cd_v2.ipynb`, `cd_iterative.ipynb`, `CD_FindCatalyst.ipynb`)  
   Use coordinate descent to nudge catalyst descriptors toward higher predicted ee, then retrieve the nearest real catalyst from the dataset via k-nearest neighbors (KNN). This creates an active-learning-style loop: optimize → find nearest candidate → add to training set → optionally retrain.

5. **Full-feature modeling** (`large/eda.ipynb`, `052124_feature_selection.ipynb`)  
   Train PLS on the full descriptor matrix after selecting highly-correlated columns (`models/high_corr_cols.txt`).

## Key Models

| Model | Input | Output | File |
|-------|-------|--------|------|
| PLS (n_components=2) | `x`, `y`, `z` | `ddG (% ee)` | `models/pls.joblib` |
| PLS (n_components=4) | Full/highly-correlated descriptors | `ddG` | `models/pls_large.joblib` |

Example baseline metrics on the 3D reduced space (30% test split, random_state=101):

- **OLS**: MSE ≈ 0.107, MAE ≈ 0.276
- **PLS (2 components)**: MSE ≈ 0.106, MAE ≈ 0.275

## Running the Project

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate ml2
jupyter lab
```

### Option 2: pip

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter lab
```

> **Note:** `requirements.txt` and `environment.yml` were exported as UTF-16-LE. If your editor or CI reports encoding issues, re-save them as UTF-8.

### Open in Colab

Several notebooks include a Colab badge and `resolve_path_gdrive()` helper to load data from Google Drive when running on Google Colab.

## Main Notebooks at a Glance

| Notebook | What it does |
|----------|--------------|
| `linear-regression.ipynb` | Train and evaluate OLS/PLS baselines on the reduced 3D space. |
| `knn.ipynb` | k-nearest-neighbor analysis on catalyst embeddings. |
| `cd_v2.ipynb` | Configurable coordinate-descent optimization loop with model retraining strategies. |
| `cd_iterative.ipynb` | Iterative version of the coordinate-descent/KNN search. |
| `CD_FindCatalyst.ipynb` | End-to-end script to optimize and find high-ee catalysts. |
| `052024_model_suite.ipynb` | Benchmark of multiple regression approaches. |
| `052124_feature_selection.ipynb` | Feature selection on the full descriptor matrix. |
| `Mordred_Descriptors.ipynb` | Generate descriptors with Mordred/RDKit. |

## Dependencies (high-level)

- Python 3.11
- `numpy`, `pandas`, `scikit-learn`, `scipy`, `statsmodels`
- `torch`, `pytorch-lightning`
- `rdkit`, `mordred` (descriptor generation)
- `jupyter`, `jupyterlab`
- `joblib`, `matplotlib`, `factor-analyzer`, `lifelines`

See `requirements.txt` / `environment.yml` for exact pinned versions.

## Notes & Caveats

- This repository is experimental and notebook-driven.
- Many notebooks assume data files live relative to the repository root (e.g., `data/reduced_dim_space_ddG.csv`) or use a Google Drive path when running in Colab.
- `environment.yml` contains a hard-coded Windows prefix (`C:\work\workspaces\python\miniconda3\envs\ml2`); you may want to remove or update `prefix:` when recreating the environment on another machine.

## License

No license is specified in the repository. If you plan to share or reuse this code, consider adding a `LICENSE` file.
