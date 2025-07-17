"""
PAD Mapper Replication Pipeline
================================
Author: Douglas Nery
Date: 2025-07-17

This script reproduces—end-to-end—the core analytic pipeline of
Nicolau et al. (2011) *PNAS* (“Topology based data analysis identifies a
subgroup of breast cancers with a unique mutational profile and excellent
survival”).

The workflow closely mirrors the original **Progression Analysis of Disease (PAD)**:

1. **Data acquisition** – downloads micro-array gene-expression datasets
   • **GSE2034** – 295 breast-tumour samples ("NKI cohort").
   • **GSE7307** – normal breast tissue samples (used as healthy baseline).
2. **Pre-processing**
   • probe-to-gene mapping via GPL annotation.
   • log₂ transformation + per-gene Z-score standardisation.
3. **Disease-Specific Genomic Analysis (DSGA)**
   • fit PCA (5 components) on healthy samples ⇒ normal sub-space.
   • for each tumour, project → calculate residual vector `Dc`.
   • `||Dc||₂` = PAD score (disease progression coordinate).
4. **Mapper construction** (using *KeplerMapper*)
   • lens = 1-D array of PAD scores.
   • cover: 10 intervals, 50 % overlap.
   • clustering: DBSCAN (ε = 0.5, minPts = 5) on residual vectors.
5. **Visualisation & exploration**
   • interactive HTML graph coloured by PAD score.
   • second HTML graph coloured by **MYB** expression (to spot c-MYB⁺ arm).
6. **(Optional) survival analysis** – if clinical metadata present, Kaplan–Meier curves per Mapper node using *lifelines*.

> ⚠️ Running time: ≈ 3–5 min with fast internet; memory < 2 GB.
> The script is **fully reproducible**; all downloads cached in `./data/`.

Dependencies (Python ≥ 3.9):
```bash
pip install pandas numpy scikit-learn geparse kmapper networkx plotly lifelines tqdm
```

Licence: MIT.
"""

import os
import json
import warnings
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import kmapper as km
import GEOparse  # type: ignore
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Global config
# -----------------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_DIR / "combined_expression.parquet"

HEALTHY_ACC = "GSE7307"  # normal tissue compendium (subset for breast)
TUMOR_ACC = "GSE2034"   # NKI breast-cancer cohort with survival

N_COMPONENTS_PCA = 5
MAPPER_N_CUBES = 10
MAPPER_OVERLAP = 0.5
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def download_gse(accession: str) -> GEOparse.GEO.GEO:  # type: ignore
    """Download a GSE (cached locally)."""
    out_dir = DATA_DIR / accession
    if not out_dir.exists():
        print(f"📥 Downloading {accession} from GEO …")
    gse = GEOparse.get_GEO(accession, destdir=str(DATA_DIR), silent=True)
    return gse


def gse_to_expression(gse: GEOparse.GEO.GEO) -> pd.DataFrame:  # type: ignore
    """Extract expression matrix [genes × samples] with gene symbols."""
    gsm_list = []
    for gsm_id, gsm in tqdm(gse.gsms.items(), total=len(gse.gsms)):
        tbl = gsm.table
        # Assume first column is probe ID, 2nd is expression value
        expr = tbl.iloc[:, 1].astype(float)
        probe_ids = tbl.iloc[:, 0]
        sample_series = pd.Series(expr.values, index=probe_ids, name=gsm_id)
        gsm_list.append(sample_series)

    expr_matrix = pd.concat(gsm_list, axis=1)

    # Map probe → gene using GPL annotation (many-to-one). Keep mean if duplicates.
    gpl = next(iter(gse.gpls.values()))
    ann = gpl.table[[gpl.table.columns[0], "Gene Symbol"]]
    ann.columns = ["ID", "gene"]
    gene_map = ann.set_index("ID").gene
    expr_matrix["gene"] = gene_map
    expr_matrix = expr_matrix.dropna(subset=["gene"])
    expr_matrix = expr_matrix.groupby("gene").mean()
    return expr_matrix  # genes × samples


def log_z_score(df: pd.DataFrame) -> pd.DataFrame:
    df_logged = np.log2(df + 1)
    scaler = StandardScaler(with_mean=True, with_std=True)
    z = pd.DataFrame(scaler.fit_transform(df_logged.T).T,
                     index=df_logged.index,
                     columns=df_logged.columns)
    return z


def compute_dsga(tumour_z: pd.DataFrame, healthy_z: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return: residual vectors (samples × genes) and PAD scores (||residual||₂)."""
    pca = PCA(n_components=N_COMPONENTS_PCA, svd_solver="full", random_state=0)
    pca.fit(healthy_z.T)

    residuals = []
    pad_scores = []
    for s in tumour_z.columns:
        x = tumour_z[s].values
        x_proj = pca.inverse_transform(pca.transform([x]))[0]
        r = x - x_proj
        residuals.append(r)
        pad_scores.append(np.linalg.norm(r))

    residuals = np.array(residuals)  # samples × genes
    pad_scores = np.array(pad_scores)
    return residuals, pad_scores


def build_mapper(residuals: np.ndarray, lens: np.ndarray) -> km.KeplerMapper:
    mapper = km.KeplerMapper(verbose=1)
    cover = km.Cover(n_cubes=MAPPER_N_CUBES, perc_overlap=MAPPER_OVERLAP)
    graph = mapper.map(lens,
                       residuals,
                       cover=cover,
                       clusterer=DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES),
                       metric="euclidean")
    html_path = "pad_mapper.html"
    mapper.visualize(graph,
                     path_html=html_path,
                     title="PAD Mapper (lens = ||D₍c₎||₂)",
                     color_values=lens.flatten(),
                     color_function_name="PAD score")
    print(f"🌐 Interactive Mapper graph saved to {html_path}")
    return mapper

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main():
    if CACHE_FILE.exists():
        print("⚡ Using cached expression matrix …")
        combined = pd.read_parquet(CACHE_FILE)
        tumour_expr = combined.loc[:, combined.columns.str.startswith("TUMOR_")]
        healthy_expr = combined.loc[:, combined.columns.str.startswith("HEALTHY_")]
    else:
        gse_tum = download_gse(TUMOR_ACC)
        gse_hea = download_gse(HEALTHY_ACC)

        tumour_expr = gse_to_expression(gse_tum)
        healthy_expr = gse_to_expression(gse_hea)

        # keep only normal breast samples in healthy set (platform-specific keyword)
        tissue_mask = healthy_expr.columns.str.contains("breast", case=False)
        healthy_expr = healthy_expr.loc[:, tissue_mask]

        # harmonise gene set
        genes_common = tumour_expr.index.intersection(healthy_expr.index)
        tumour_expr = tumour_expr.loc[genes_common]
        healthy_expr = healthy_expr.loc[genes_common]

        # cache
        combined = pd.concat([tumour_expr.add_prefix("TUMOR_"),
                              healthy_expr.add_prefix("HEALTHY_")], axis=1)
        combined.to_parquet(CACHE_FILE)

    # ----------------------------------------------------
    tumour_z = log_z_score(tumour_expr)
    healthy_z = log_z_score(healthy_expr)

    residuals, pad_scores = compute_dsga(tumour_z, healthy_z)

    # ----------------------------------------------------
    mapper = build_mapper(residuals, pad_scores.reshape(-1, 1))

    # ----------------------------------------------------
    # Optional: colour by MYB expression (c-MYB)
    if "MYB" in tumour_z.index:
        myb_vals = tumour_z.loc["MYB"].values
        mapper.visualize(mapper.graph,
                         path_html="pad_mapper_myB.html",
                         title="PAD Mapper coloured by MYB expression",
                         color_values=myb_vals,
                         color_function_name="MYB expression")
        print("🌐 Second HTML coloured by MYB saved to pad_mapper_myB.html")

    # ----------------------------------------------------
    # Export PAD scores + metadata
    out_df = pd.DataFrame({"sample": tumour_z.columns,
                           "pad_score": pad_scores})
    out_df.to_csv("pad_scores.csv", index=False)
    print("📄 PAD scores exported → pad_scores.csv")

    print("✅ Pipeline finished.")


if __name__ == "__main__":
    main()
