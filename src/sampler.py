"""
sampler.py — Single-cell sampling pipeline for DarwinSeq.

Provides CellSampler, which randomly draws n individual cells from the
full h5ad dataset, retrieves their expression vectors for a specified gene
list, and pairs them with the selected prediction target.

Design notes:
- Sampling unit is a single cell (not pseudobulk). The SEAAD A9 dataset
  has ~84 unique donors, which is fewer than the default n=100 sample size,
  making pseudobulk aggregation infeasible for n=100.
- ADNC labels (y) are donor-level values stored in obs. All cells from
  the same donor share identical y values.
- For the full pipeline, gene expression is loaded via DataLoader's
  h5py-based chunked reader, keeping memory usage low.
"""

import numpy as np
import pandas as pd

from src.data_loader import DataLoader


class CellSampler:
    """
    Randomly sample n cells and return expression + prediction target matrices.

    Args:
        data_loader: A DataLoader instance (or any object with the same
                     interface: .adata.obs_names, .get_expression_for_cells(),
                     .get_adnc_target(), .get_pathology_targets()).
        gene_list: Ordered list of gene symbols to extract expression for.
        seed: Random seed for reproducibility. None means non-deterministic.
        target: Which target to return with y.
                "adnc"      — ADNC ordinal class (default); y has one column.
                "pathology" — legacy 6-column continuous pathology targets.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        gene_list: list[str],
        seed: int | None = None,
        target: str = "adnc",
    ) -> None:
        if not gene_list:
            raise ValueError("gene_list must not be empty.")
        if target not in ("adnc", "pathology"):
            raise ValueError(
                f"target must be 'adnc' or 'pathology', got '{target}'."
            )
        self.data_loader = data_loader
        self.gene_list = list(gene_list)
        self.seed = seed
        self.target = target

    def sample(self, n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Randomly sample n cells and return their expression and prediction target.

        Sampling is performed without replacement. The same seed always
        produces the same cell selection.

        Args:
            n: Number of cells to sample. Must not exceed the total cell count
               in data_loader.adata.

        Returns:
            X: DataFrame of shape (n, len(gene_list)) with float32 expression
               values. Index is cell barcodes, columns are gene_list.
            y: DataFrame whose content depends on self.target:
                 - "adnc"      → shape (n, 1), column "ADNC", float (0–3 or NaN).
                 - "pathology" → shape (n, 6), one column per pathology target.
               Index matches X. Values may be NaN where data is missing.

        Raises:
            ValueError: If n exceeds the number of available cells.
        """
        obs_names = list(self.data_loader.adata.obs_names)
        n_available = len(obs_names)

        if n > n_available:
            raise ValueError(
                f"Requested {n} cells but only {n_available} are available."
            )

        rng = np.random.default_rng(self.seed)
        chosen_idx = rng.choice(n_available, size=n, replace=False)
        barcodes = [obs_names[i] for i in chosen_idx]

        X = self.data_loader.get_expression_for_cells(barcodes, self.gene_list)

        if self.target == "adnc":
            y = self.data_loader.get_adnc_target(barcodes).to_frame()
        else:
            y = self.data_loader.get_pathology_targets(barcodes)

        return X, y
