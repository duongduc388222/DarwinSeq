"""
sampler.py — Single-cell sampling pipeline for DarwinSeq.

Provides CellSampler, which randomly draws n individual cells from the
full h5ad dataset, retrieves their expression vectors for a specified gene
list, and pairs them with donor-level pathology targets.

Design notes:
- Sampling unit is a single cell (not pseudobulk). The SEAAD A9 dataset
  has ~84 unique donors, which is fewer than the default n=100 sample size,
  making pseudobulk aggregation infeasible for n=100.
- Pathology targets (y) are donor-level values stored in obs. All cells from
  the same donor share identical y values; the evaluator (milestone 4+)
  should account for this repeated-measures structure.
- For the full pipeline, gene expression is loaded via DataLoader's
  h5py-based chunked reader, keeping memory usage low.
"""

import numpy as np
import pandas as pd

from src.data_loader import DataLoader


class CellSampler:
    """
    Randomly sample n cells and return expression + pathology target matrices.

    Args:
        data_loader: A DataLoader instance (or any object with the same
                     interface: .adata.obs_names, .get_expression_for_cells(),
                     .get_pathology_targets()).
        gene_list: Ordered list of gene symbols to extract expression for.
        seed: Random seed for reproducibility. None means non-deterministic.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        gene_list: list[str],
        seed: int | None = None,
    ) -> None:
        if not gene_list:
            raise ValueError("gene_list must not be empty.")
        self.data_loader = data_loader
        self.gene_list = list(gene_list)
        self.seed = seed

    def sample(self, n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Randomly sample n cells and return their expression and pathology targets.

        Sampling is performed without replacement. The same seed always
        produces the same cell selection.

        Args:
            n: Number of cells to sample. Must not exceed the total cell count
               in data_loader.adata.

        Returns:
            X: DataFrame of shape (n, len(gene_list)) with float32 expression
               values. Index is cell barcodes, columns are gene_list.
            y: DataFrame of shape (n, 6) with pathology target values.
               Index matches X. Columns are the 6 pathology target names
               present in obs. Values may be NaN where data is missing.

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
        y = self.data_loader.get_pathology_targets(barcodes)

        return X, y
