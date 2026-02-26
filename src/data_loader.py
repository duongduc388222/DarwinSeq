"""
data_loader.py — Safe AnnData loader with corruption detection and DataLoader interface.

Provides safe_load() as the single entry point for loading .h5ad files
throughout the project. Because the SEAAD dataset is very large (~1.3M cells,
7.4B non-zero entries, ~28 GB uncompressed), loading the full matrix into
memory is not feasible. Instead:

  - obs and var are loaded fully (metadata only, fast).
  - Expression corruption checks are done via chunked h5py scanning on the
    sparse indptr/data arrays (no full in-memory load).
  - The returned AnnData contains only obs/var; the h5ad path is stored in
    adata.uns["h5ad_path"] so downstream code can open the file directly
    for chunked expression access.

All downstream milestones should import and use safe_load() instead of
calling anndata.read_h5ad() directly.

DataLoader wraps the metadata-only AnnData returned by safe_load() and
exposes methods for on-demand expression retrieval via h5py CSR reads.
It is the primary interface for the sampling pipeline (milestone 3+).
"""

import json
import os
from pathlib import Path

import hdf5plugin  # noqa: F401 — registers blosc/lzf/zstd HDF5 filters
import anndata
import h5py
import numpy as np
import pandas as pd
from anndata._io.specs import read_elem

# Default path to the SEA-AD DREAM challenge dataset.
# Pass a different path via the h5ad_path argument to override.
DEFAULT_DATA_PATH = (
    "/Users/duonghongduc/GrinnellCollege/MLAI/__CODE/ADetective/data/"
    "SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad"
)

# Pathology target columns required for downstream tasks.
_PATHOLOGY_COLS = [
    "percent 6e10 positive area",
    "percent AT8 positive area",
    "percent NeuN positive area",
    "percent GFAP positive area",
    "percent pTDP43 positive area",
    "percent aSyn positive area",
]

# Metadata columns that must not be NaN for a sample to be usable.
_REQUIRED_META_COLS = ["Donor ID"]

# Maximum tolerated corruption rate before aborting.
_MAX_DROP_RATE = 0.05

# Chunk size (number of cells) for scanning expression data via h5py.
_EXPR_CHUNK_SIZE = 50_000

# Number of randomly sampled cells for NaN/Inf/negative expression checks.
# All-zero checks run on all cells (uses only indptr, very fast).
# Set to None to check every cell (much slower for large datasets).
_EXPR_SAMPLE_SIZE = 20_000


def safe_load(
    h5ad_path: str = DEFAULT_DATA_PATH,
    report_dir: str = "data/",
) -> anndata.AnnData:
    """
    Load a .h5ad file with corruption detection and filtering.

    Because the dataset is too large to load into memory (~28 GB uncompressed),
    this function:
      1. Reads obs (cell metadata) and var (gene metadata) only — fast and
         low memory.
      2. Runs metadata corruption checks on obs.
      3. Runs expression corruption checks by scanning the sparse X matrix
         via h5py in chunks (no full load).
      4. Returns a metadata-only AnnData (obs/var, no X in memory).
         adata.uns["h5ad_path"] stores the file path so downstream code can
         open the HDF5 file directly for chunked expression access.

    Writes a corruption report JSON to report_dir/corruption_report.json.
    Raises RuntimeError if the corruption drop rate exceeds 5%.

    Args:
        h5ad_path: Path to the .h5ad file. Defaults to DEFAULT_DATA_PATH.
        report_dir: Directory where corruption_report.json will be written.

    Returns:
        Metadata-only anndata.AnnData with clean cells in obs/var.
        Use adata.uns["h5ad_path"] to open the file for expression access.

    Raises:
        FileNotFoundError: If h5ad_path does not exist.
        RuntimeError: If the corruption drop rate exceeds 5%.
    """
    h5ad_path = str(h5ad_path)
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"Data file not found: {h5ad_path}")

    print(f"Loading metadata from {h5ad_path} ...")
    with h5py.File(h5ad_path, "r") as f:
        obs: pd.DataFrame = read_elem(f["obs"])
        var: pd.DataFrame = read_elem(f["var"])

    original_n = len(obs)
    filename = Path(h5ad_path).name

    # Build a lightweight AnnData for passing to metadata check helpers.
    adata_meta = anndata.AnnData(obs=obs, var=var)

    bad_meta = _check_metadata_corruption(adata_meta)
    bad_expr = _check_expression_corruption(h5ad_path, adata_meta.obs_names)

    all_bad: dict[str, dict] = {}
    for entry in bad_meta + bad_expr:
        idx = entry["index"]
        if idx not in all_bad:
            all_bad[idx] = entry
        else:
            all_bad[idx]["reason"] += f"|{entry['reason']}"

    dropped_samples = list(all_bad.values())
    bad_indices = set(all_bad.keys())

    drop_n = len(bad_indices)
    clean_n = original_n - drop_n
    drop_rate = drop_n / original_n if original_n > 0 else 0.0

    if drop_rate > _MAX_DROP_RATE:
        raise RuntimeError(
            f"Corruption drop rate {drop_rate:.2%} exceeds {_MAX_DROP_RATE:.0%} "
            f"for {filename}. Investigate file-level integrity before proceeding."
        )

    # Build cleaned metadata AnnData.
    if bad_indices:
        keep_mask = ~adata_meta.obs_names.isin(bad_indices)
        adata_clean = anndata.AnnData(
            obs=obs[keep_mask.values],
            var=var,
        )
    else:
        adata_clean = adata_meta

    # Store path so downstream code can access X via h5py without re-reading metadata.
    adata_clean.uns["h5ad_path"] = h5ad_path

    # Write corruption report.
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "corruption_report.json")
    report = {
        "file": filename,
        "original_n_samples": original_n,
        "dropped_n_samples": drop_n,
        "clean_n_samples": clean_n,
        "drop_rate_percent": round(drop_rate * 100, 4),
        "dropped_samples": dropped_samples,
    }
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(
        f"Loaded {filename}: {original_n:,} → {clean_n:,} cells "
        f"(dropped {drop_n:,}, {drop_rate:.3%})"
    )
    return adata_clean


def _check_expression_corruption(
    h5ad_path: str,
    obs_names: "pd.Index",
) -> list[dict]:
    """
    Check the sparse X matrix for all-zero rows (via indptr, O(n_cells)) and
    for NaN / Inf / negative values on a random sample of cells.

    Reads data via h5py in chunks — no full matrix is loaded into memory.

    Args:
        h5ad_path: Path to the .h5ad file.
        obs_names: Cell barcodes in row order (used to map row indices to names).

    Returns:
        List of dicts, each with keys: index, reason, details.
    """
    bad: list[dict] = []
    n_cells = len(obs_names)

    with h5py.File(h5ad_path, "r") as f:
        indptr = f["X/indptr"][:]  # shape (n_cells + 1,) — fast, ~10 MB

        # --- All-zero check: O(n_cells), uses only indptr ---
        row_nnz = np.diff(indptr)  # number of non-zeros per cell
        zero_rows = np.where(row_nnz == 0)[0]
        for r in zero_rows:
            bad.append({
                "index": obs_names[r],
                "reason": "all_zero_expression",
                "details": "entire expression vector is zero (empty/dead cell)",
            })

        # --- NaN / Inf / negative: scan a random sample to avoid reading 28 GB ---
        sample_size = min(_EXPR_SAMPLE_SIZE, n_cells)
        rng = np.random.default_rng(seed=42)
        sampled_rows = np.sort(rng.choice(n_cells, size=sample_size, replace=False))

        # Group sampled rows into contiguous chunks for efficient h5py slicing.
        for chunk_start_idx in range(0, len(sampled_rows), _EXPR_CHUNK_SIZE):
            chunk_rows = sampled_rows[chunk_start_idx: chunk_start_idx + _EXPR_CHUNK_SIZE]

            for r in chunk_rows:
                ds = int(indptr[r])
                de = int(indptr[r + 1])
                if ds == de:
                    continue  # already caught by all-zero check
                values = f["X/data"][ds:de]

                if np.any(np.isnan(values)):
                    count = int(np.sum(np.isnan(values)))
                    bad.append({
                        "index": obs_names[r],
                        "reason": "nan_in_expression",
                        "details": f"{count} NaN value(s) in expression vector",
                    })
                elif np.any(np.isinf(values)):
                    count = int(np.sum(np.isinf(values)))
                    bad.append({
                        "index": obs_names[r],
                        "reason": "inf_in_expression",
                        "details": f"{count} Inf value(s) in expression vector",
                    })
                elif np.any(values < 0):
                    count = int(np.sum(values < 0))
                    bad.append({
                        "index": obs_names[r],
                        "reason": "negative_expression",
                        "details": f"{count} negative value(s) in expression vector",
                    })

    return bad


def _check_metadata_corruption(adata: anndata.AnnData) -> list[dict]:
    """
    Check adata.obs for missing pathology targets, required metadata NaNs,
    and duplicate cell barcodes.

    Args:
        adata: AnnData object (metadata only is sufficient).

    Returns:
        List of dicts, each with keys: index, reason, details.
    """
    bad: list[dict] = []
    obs = adata.obs
    obs_names = adata.obs_names

    # Missing pathology targets: flag rows where ALL columns are NaN.
    present_pathology = [c for c in _PATHOLOGY_COLS if c in obs.columns]
    if present_pathology:
        all_nan_mask = obs[present_pathology].isna().all(axis=1)
        for idx in obs_names[all_nan_mask]:
            bad.append({
                "index": idx,
                "reason": "missing_all_pathology_targets",
                "details": f"all of {present_pathology} are NaN",
            })

    # Required metadata columns NaN.
    for col in _REQUIRED_META_COLS:
        if col not in obs.columns:
            continue
        nan_mask = obs[col].isna()
        for idx in obs_names[nan_mask]:
            bad.append({
                "index": idx,
                "reason": f"missing_metadata_{col}",
                "details": f"column '{col}' is NaN",
            })

    # Duplicate barcodes.
    seen: set[str] = set()
    for idx in obs_names:
        if idx in seen:
            bad.append({
                "index": idx,
                "reason": "duplicate_barcode",
                "details": f"cell barcode '{idx}' appears more than once",
            })
        seen.add(idx)

    return bad


class DataLoader:
    """
    Unified data loading interface wrapping a metadata-only AnnData.

    Calls safe_load() once to obtain clean obs/var metadata, then provides
    on-demand expression extraction via h5py CSR reads. Expression is never
    loaded in bulk — only the rows and columns requested are read from disk.

    Args:
        h5ad_path: Path to the .h5ad file. Defaults to DEFAULT_DATA_PATH.
        report_dir: Directory for the corruption_report.json. Defaults to "data/".
    """

    def __init__(
        self,
        h5ad_path: str = DEFAULT_DATA_PATH,
        report_dir: str = "data/",
    ) -> None:
        self.adata = safe_load(h5ad_path, report_dir)
        self.h5ad_path: str = self.adata.uns["h5ad_path"]

        # Read the original (pre-filter) barcode order from the h5py file so
        # we can map any clean barcode to its correct row index in X.
        with h5py.File(self.h5ad_path, "r") as f:
            orig_obs: pd.DataFrame = read_elem(f["obs"])
            orig_var: pd.DataFrame = read_elem(f["var"])

        self._barcode_to_row: dict[str, int] = {
            bc: i for i, bc in enumerate(orig_obs.index)
        }
        self._gene_names: list[str] = list(orig_var.index)
        # Upper-case key → h5py column index (enables case-insensitive lookup).
        self._gene_upper_to_col: dict[str, int] = {
            g.upper(): i for i, g in enumerate(self._gene_names)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_expression_for_cells(
        self,
        cell_barcodes: list[str],
        gene_list: list[str],
    ) -> pd.DataFrame:
        """
        Load expression values for specific cells and genes via h5py CSR reads.

        Reads only the requested rows and columns — no full matrix load.
        Gene name matching is case-insensitive. Genes absent from the dataset
        are returned as all-zeros with a logged warning.

        Args:
            cell_barcodes: List of cell barcodes to extract.
            gene_list: List of gene symbols to extract.

        Returns:
            DataFrame of shape (len(cell_barcodes), len(gene_list)) with
            float32 expression values. Index is cell_barcodes, columns are gene_list.

        Raises:
            KeyError: If any barcode is not found in the h5ad file.
        """
        # Resolve gene names → h5py column indices (-1 if not found).
        col_indices = np.array(
            [self._gene_upper_to_col.get(g.upper(), -1) for g in gene_list],
            dtype=np.int64,
        )
        missing_genes = [g for g, c in zip(gene_list, col_indices) if c < 0]
        if missing_genes:
            import logging
            logging.getLogger(__name__).warning(
                "%d gene(s) not found in h5ad, returning zeros: %s",
                len(missing_genes),
                missing_genes[:5],
            )

        # Resolve barcodes → h5py row indices.
        try:
            row_indices = np.array(
                [self._barcode_to_row[bc] for bc in cell_barcodes],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise KeyError(f"Barcode not found in h5ad: {exc}") from exc

        n_cells = len(cell_barcodes)
        n_genes = len(gene_list)
        result = np.zeros((n_cells, n_genes), dtype=np.float32)

        # Indices of genes that actually exist (skip -1 entries).
        valid_j = np.where(col_indices >= 0)[0]
        valid_cols = col_indices[valid_j]

        with h5py.File(self.h5ad_path, "r") as f:
            indptr = f["X/indptr"][:]  # load once — ~10 MB for 1.3M cells

            for i, row_idx in enumerate(row_indices):
                ds = int(indptr[row_idx])
                de = int(indptr[row_idx + 1])
                if ds == de:
                    continue  # all-zero row

                row_cols = f["X/indices"][ds:de]  # sorted within CSR row
                row_vals = f["X/data"][ds:de]

                # Vectorised binary-search for all requested gene columns.
                pos = np.searchsorted(row_cols, valid_cols)
                found = (pos < len(row_cols)) & (row_cols[pos] == valid_cols)
                result[i, valid_j[found]] = row_vals[pos[found]]

        return pd.DataFrame(result, index=cell_barcodes, columns=gene_list)

    def get_pathology_targets(
        self,
        cell_barcodes: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return the 6 pathology target columns from obs.

        Args:
            cell_barcodes: If provided, return only rows for these barcodes.
                           If None, return all cells.

        Returns:
            DataFrame with columns matching _PATHOLOGY_COLS (those present in obs).
            Missing columns are silently excluded.
        """
        present = [c for c in _PATHOLOGY_COLS if c in self.adata.obs.columns]
        obs_sub = (
            self.adata.obs.loc[cell_barcodes, present]
            if cell_barcodes is not None
            else self.adata.obs[present]
        )
        return obs_sub.copy()

    def get_metadata(
        self,
        cell_barcodes: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return all obs metadata columns.

        Args:
            cell_barcodes: If provided, return only rows for these barcodes.
                           If None, return all cells.

        Returns:
            Full obs DataFrame (or subset for requested barcodes).
        """
        if cell_barcodes is not None:
            return self.adata.obs.loc[cell_barcodes].copy()
        return self.adata.obs.copy()
