"""
test_sampler.py — Unit tests for src/sampler.py.

All tests use FakeDataLoader — a lightweight stub that satisfies the
DataLoader interface without touching the real h5ad file.
"""

import numpy as np
import pandas as pd
import pytest

from src.sampler import CellSampler


# ---------------------------------------------------------------------------
# Stub DataLoader
# ---------------------------------------------------------------------------

_PATHOLOGY_COLS = [
    "percent 6e10 positive area",
    "percent AT8 positive area",
    "percent NeuN positive area",
    "percent GFAP positive area",
    "percent pTDP43 positive area",
    "percent aSyn positive area",
]


class _FakeAdata:
    """Minimal stand-in for anndata.AnnData with obs only."""

    def __init__(self, obs: pd.DataFrame) -> None:
        self.obs = obs
        self.obs_names = obs.index


class FakeDataLoader:
    """
    Stub that satisfies the DataLoader interface used by CellSampler.

    Expression values are deterministic floats derived from the barcode
    index and gene position, making assertions straightforward.
    """

    def __init__(self, n_cells: int = 500, n_genes: int = 300) -> None:
        self.n_cells = n_cells
        self.n_genes = n_genes
        self._gene_names = [f"GENE_{i}" for i in range(n_genes)]
        self._gene_to_idx = {g: i for i, g in enumerate(self._gene_names)}

        barcodes = [f"CELL_{i:04d}" for i in range(n_cells)]

        # Donor assignments: cycle through 80 donors.
        n_donors = 80
        donors = [f"DONOR_{i % n_donors:03d}" for i in range(n_cells)]

        # Pathology targets: deterministic values (donor index / n_donors).
        rng = np.random.default_rng(0)
        pathology_data = rng.uniform(0, 1, size=(n_cells, 6))
        obs = pd.DataFrame(
            pathology_data,
            index=barcodes,
            columns=_PATHOLOGY_COLS,
        )
        obs["Donor ID"] = donors
        # ADNC labels: cycle through 4 classes (0–3) deterministically.
        obs["ADNC"] = [float(i % 4) for i in range(n_cells)]

        self.adata = _FakeAdata(obs)

    def get_expression_for_cells(
        self, cell_barcodes: list[str], gene_list: list[str]
    ) -> pd.DataFrame:
        """Return deterministic float values for requested cells × genes."""
        cell_idx = np.array(
            [int(bc.split("_")[1]) for bc in cell_barcodes], dtype=float
        )
        gene_idx = np.array(
            [self._gene_to_idx.get(g, 0) for g in gene_list], dtype=float
        )
        # Expression[i, j] = (cell_index + 1) * (gene_index + 1) * 0.001
        expr = np.outer(cell_idx + 1, gene_idx + 1) * 0.001
        return pd.DataFrame(
            expr.astype(np.float32), index=cell_barcodes, columns=gene_list
        )

    def get_adnc_target(
        self, cell_barcodes: list[str] | None = None
    ) -> pd.Series:
        """
        Return ADNC integer-encoded labels for requested barcodes.

        Args:
            cell_barcodes: List of barcode strings to look up, or None for all.

        Returns:
            pd.Series named "ADNC" with float values 0.0–3.0.
        """
        series = self.adata.obs["ADNC"]
        if cell_barcodes is not None:
            return series.loc[cell_barcodes].copy()
        return series.copy()

    def get_donor_ids(
        self, cell_barcodes: list[str] | None = None
    ) -> pd.Series:
        """Return Donor ID strings for requested barcodes."""
        series = self.adata.obs["Donor ID"]
        if cell_barcodes is not None:
            return series.loc[cell_barcodes].copy()
        return series.copy()

    def get_pathology_targets(
        self, cell_barcodes: list[str] | None = None
    ) -> pd.DataFrame:
        """Return pathology obs columns for requested barcodes."""
        obs = self.adata.obs[_PATHOLOGY_COLS]
        if cell_barcodes is not None:
            return obs.loc[cell_barcodes].copy()
        return obs.copy()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loader():
    """FakeDataLoader with 500 cells, 300 genes."""
    return FakeDataLoader(n_cells=500, n_genes=300)


@pytest.fixture
def gene_list():
    """200 genes: first 200 from the FakeDataLoader's gene space."""
    return [f"GENE_{i}" for i in range(200)]


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_sample_x_shape(loader, gene_list):
    """X should be (100, 200)."""
    sampler = CellSampler(loader, gene_list, seed=42)
    X, _ = sampler.sample(n=100)
    assert X.shape == (100, 200)


def test_sample_y_shape(loader, gene_list):
    """y (adnc mode) should be (100, 2) with columns ['ADNC', 'Donor ID']."""
    sampler = CellSampler(loader, gene_list, seed=42)
    _, y = sampler.sample(n=100)
    assert y.shape == (100, 2)
    assert list(y.columns) == ["ADNC", "Donor ID"]


def test_sample_x_columns_match_gene_list(loader, gene_list):
    """X columns must exactly match the requested gene_list in order."""
    sampler = CellSampler(loader, gene_list, seed=42)
    X, _ = sampler.sample(n=100)
    assert list(X.columns) == gene_list


def test_sample_y_columns_are_pathology_targets(loader, gene_list):
    """y columns must be the pathology target names (legacy pathology mode)."""
    sampler = CellSampler(loader, gene_list, seed=42, target="pathology")
    _, y = sampler.sample(n=100)
    assert list(y.columns) == _PATHOLOGY_COLS


# ---------------------------------------------------------------------------
# Index / barcode tests
# ---------------------------------------------------------------------------


def test_sample_index_shared(loader, gene_list):
    """X and y must share the same index (cell barcodes)."""
    sampler = CellSampler(loader, gene_list, seed=42)
    X, y = sampler.sample(n=100)
    assert list(X.index) == list(y.index)


def test_sample_no_duplicate_barcodes(loader, gene_list):
    """Sampled cells must be unique (no replacement)."""
    sampler = CellSampler(loader, gene_list, seed=42)
    X, _ = sampler.sample(n=100)
    assert len(X.index) == len(set(X.index))


def test_sampled_barcodes_exist_in_loader(loader, gene_list):
    """All returned barcodes must come from data_loader.adata.obs_names."""
    sampler = CellSampler(loader, gene_list, seed=42)
    X, _ = sampler.sample(n=100)
    all_barcodes = set(loader.adata.obs_names)
    assert all(bc in all_barcodes for bc in X.index)


# ---------------------------------------------------------------------------
# Expression value tests
# ---------------------------------------------------------------------------


def test_expression_values_are_float32(loader, gene_list):
    """Expression values in X should be float32."""
    sampler = CellSampler(loader, gene_list, seed=42)
    X, _ = sampler.sample(n=100)
    assert X.dtypes.unique()[0] == np.float32


def test_expression_values_match_stub(loader, gene_list):
    """Expression values should match what FakeDataLoader would return directly."""
    sampler = CellSampler(loader, gene_list, seed=7)
    X, _ = sampler.sample(n=10)

    # Re-fetch directly from the loader and compare.
    barcodes = list(X.index)
    expected = loader.get_expression_for_cells(barcodes, gene_list)
    pd.testing.assert_frame_equal(X, expected)


def test_pathology_targets_match_obs(loader, gene_list):
    """y values (pathology mode) should match pathology obs columns."""
    sampler = CellSampler(loader, gene_list, seed=7, target="pathology")
    _, y = sampler.sample(n=10)

    barcodes = list(y.index)
    expected = loader.adata.obs.loc[barcodes, _PATHOLOGY_COLS]
    pd.testing.assert_frame_equal(y, expected)


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


def test_same_seed_reproducible(loader, gene_list):
    """Same seed must produce identical X and y."""
    sampler1 = CellSampler(loader, gene_list, seed=99)
    sampler2 = CellSampler(loader, gene_list, seed=99)
    X1, y1 = sampler1.sample(n=50)
    X2, y2 = sampler2.sample(n=50)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_frame_equal(y1, y2)


def test_different_seeds_differ(loader, gene_list):
    """Different seeds should (almost certainly) produce different samples."""
    sampler1 = CellSampler(loader, gene_list, seed=1)
    sampler2 = CellSampler(loader, gene_list, seed=2)
    X1, _ = sampler1.sample(n=50)
    X2, _ = sampler2.sample(n=50)
    assert list(X1.index) != list(X2.index)


# ---------------------------------------------------------------------------
# Edge-case / error tests
# ---------------------------------------------------------------------------


def test_n_exceeds_available_raises(loader, gene_list):
    """Requesting more cells than available should raise ValueError."""
    sampler = CellSampler(loader, gene_list, seed=0)
    with pytest.raises(ValueError, match="available"):
        sampler.sample(n=loader.n_cells + 1)


def test_empty_gene_list_raises():
    """An empty gene_list should raise ValueError at construction."""
    loader = FakeDataLoader()
    with pytest.raises(ValueError, match="gene_list"):
        CellSampler(loader, gene_list=[], seed=0)


def test_small_n_works(loader, gene_list):
    """Sampling n=1 should work and return single-row DataFrames."""
    sampler = CellSampler(loader, gene_list, seed=0)
    X, y = sampler.sample(n=1)
    assert X.shape == (1, 200)
    assert y.shape == (1, 2)  # adnc mode: ['ADNC', 'Donor ID']


def test_n_equals_all_cells(loader, gene_list):
    """Sampling all available cells should work without errors."""
    sampler = CellSampler(loader, gene_list, seed=0)
    X, y = sampler.sample(n=loader.n_cells)
    assert X.shape[0] == loader.n_cells
    assert len(X.index) == len(set(X.index))  # no duplicates


def test_default_n_is_100(loader, gene_list):
    """Default n should be 100."""
    sampler = CellSampler(loader, gene_list, seed=0)
    X, _ = sampler.sample()
    assert X.shape[0] == 100
