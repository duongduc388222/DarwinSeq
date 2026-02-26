"""
test_baseline.py — Unit tests for scripts/run_baseline.py.

All tests use FakeDataLoader and FakeVocab stubs so no real h5ad file is needed.
The three importable functions tested are:
  - run_single_seed()
  - compute_summary()
  - select_best_run()
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow importing from project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_baseline import compute_summary, run_single_seed, select_best_run
from src.evaluator import ADNCEvaluator as LASSOEvaluator, DEFAULT_CONFIG_PATH


# ---------------------------------------------------------------------------
# Stubs
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
    """Minimal AnnData stand-in with obs only."""

    def __init__(self, obs: pd.DataFrame) -> None:
        self.obs = obs
        self.obs_names = obs.index


class FakeDataLoader:
    """
    Stub satisfying the DataLoader interface used by CellSampler.

    Expression = outer product of cell index and gene index × 0.001.
    Pathology targets are random but seeded, so reproducible.
    """

    def __init__(self, n_cells: int = 300, n_genes: int = 400) -> None:
        self.n_cells = n_cells
        self.n_genes = n_genes
        self._gene_names = [f"GENE_{i}" for i in range(n_genes)]
        self._gene_to_idx = {g: i for i, g in enumerate(self._gene_names)}

        barcodes = [f"CELL_{i:04d}" for i in range(n_cells)]
        rng = np.random.default_rng(0)
        pathology_data = rng.uniform(0, 1, size=(n_cells, 6))
        obs = pd.DataFrame(pathology_data, index=barcodes, columns=_PATHOLOGY_COLS)
        obs["Donor ID"] = [f"DONOR_{i % 80:03d}" for i in range(n_cells)]
        self.adata = _FakeAdata(obs)

    def get_expression_for_cells(
        self, cell_barcodes: list[str], gene_list: list[str]
    ) -> pd.DataFrame:
        """Deterministic float expression values."""
        cell_idx = np.array([int(bc.split("_")[1]) for bc in cell_barcodes], dtype=float)
        gene_idx = np.array([self._gene_to_idx.get(g, 0) for g in gene_list], dtype=float)
        expr = np.outer(cell_idx + 1, gene_idx + 1) * 0.001
        return pd.DataFrame(expr.astype(np.float32), index=cell_barcodes, columns=gene_list)

    def get_pathology_targets(
        self, cell_barcodes: list[str] | None = None
    ) -> pd.DataFrame:
        """Return pathology obs columns for requested barcodes."""
        obs = self.adata.obs[_PATHOLOGY_COLS]
        if cell_barcodes is not None:
            return obs.loc[cell_barcodes].copy()
        return obs.copy()

    def get_adnc_target(
        self, cell_barcodes: list[str] | None = None
    ) -> pd.Series:
        """Return ADNC ordinal labels (0–3) as a float Series for requested barcodes."""
        all_barcodes = list(self.adata.obs_names)
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 4, size=self.n_cells).astype(float)
        label_series = pd.Series(labels, index=all_barcodes, name="ADNC")
        if cell_barcodes is not None:
            return label_series.loc[cell_barcodes]
        return label_series

    def get_donor_ids(
        self, cell_barcodes: list[str] | None = None
    ) -> pd.Series:
        """Return Donor ID strings for requested barcodes."""
        series = self.adata.obs["Donor ID"]
        if cell_barcodes is not None:
            return series.loc[cell_barcodes].copy()
        return series.copy()


class FakeVocab:
    """
    Stub satisfying the GeneVocabulary interface.

    in_vocab  = GENE_0 … GENE_199  (200 genes)
    out_vocab = GENE_200 … GENE_399 (200 genes)
    sample_subset(n_in, n_out, seed) returns n_in + n_out gene names.
    """

    def __init__(self) -> None:
        self.in_vocab = [f"GENE_{i}" for i in range(200)]
        self.out_vocab = [f"GENE_{i}" for i in range(200, 400)]

    def sample_subset(
        self, n_in: int = 100, n_out: int = 100, seed: int = None
    ) -> list[str]:
        """Return n_in in-vocab and n_out out-of-vocab genes."""
        rng = np.random.default_rng(seed)
        sampled_in = rng.choice(self.in_vocab, size=n_in, replace=False).tolist()
        sampled_out = rng.choice(self.out_vocab, size=n_out, replace=False).tolist()
        return sampled_in + sampled_out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loader():
    return FakeDataLoader(n_cells=300, n_genes=400)


@pytest.fixture
def vocab():
    return FakeVocab()


@pytest.fixture
def evaluator():
    return LASSOEvaluator(config_path=DEFAULT_CONFIG_PATH)


# ---------------------------------------------------------------------------
# run_single_seed tests
# ---------------------------------------------------------------------------


def test_run_single_seed_required_keys(loader, vocab, evaluator):
    """Result dict must contain all required keys."""
    result = run_single_seed(loader, vocab, evaluator, seed=0, n_in=10, n_out=10, n_cells=50)
    required = {"seed", "gene_list", "balanced_accuracy", "aggregate_score", "retained_genes", "n_retained"}
    assert required.issubset(result.keys())


def test_run_single_seed_gene_list_length(loader, vocab, evaluator):
    """gene_list length must equal n_in + n_out."""
    result = run_single_seed(loader, vocab, evaluator, seed=0, n_in=10, n_out=15, n_cells=50)
    assert len(result["gene_list"]) == 25


def test_run_single_seed_seed_stored(loader, vocab, evaluator):
    """The seed value in the result must match what was passed."""
    result = run_single_seed(loader, vocab, evaluator, seed=77, n_in=10, n_out=10, n_cells=50)
    assert result["seed"] == 77


def test_run_single_seed_n_retained_consistent(loader, vocab, evaluator):
    """n_retained must equal len(retained_genes)."""
    result = run_single_seed(loader, vocab, evaluator, seed=0, n_in=10, n_out=10, n_cells=50)
    assert result["n_retained"] == len(result["retained_genes"])


def test_run_single_seed_retained_genes_subset_of_gene_list(loader, vocab, evaluator):
    """retained_genes must be a subset of gene_list."""
    result = run_single_seed(loader, vocab, evaluator, seed=0, n_in=10, n_out=10, n_cells=50)
    assert set(result["retained_genes"]).issubset(set(result["gene_list"]))


def test_run_single_seed_determinism(loader, vocab, evaluator):
    """Same seed must produce identical results (NaN == NaN treated as equal)."""
    r1 = run_single_seed(loader, vocab, evaluator, seed=5, n_in=10, n_out=10, n_cells=50)
    r2 = run_single_seed(loader, vocab, evaluator, seed=5, n_in=10, n_out=10, n_cells=50)
    assert r1["gene_list"] == r2["gene_list"]
    assert r1["retained_genes"] == r2["retained_genes"]
    # Use numpy to compare aggregate_score so NaN == NaN passes.
    np.testing.assert_equal(r1["aggregate_score"], r2["aggregate_score"])


def test_run_single_seed_different_seeds_differ(loader, vocab, evaluator):
    """Different seeds should produce different gene lists."""
    r1 = run_single_seed(loader, vocab, evaluator, seed=1, n_in=50, n_out=50, n_cells=50)
    r2 = run_single_seed(loader, vocab, evaluator, seed=2, n_in=50, n_out=50, n_cells=50)
    assert r1["gene_list"] != r2["gene_list"]


# ---------------------------------------------------------------------------
# compute_summary tests
# ---------------------------------------------------------------------------


def _make_runs(n: int, base_score: float = 0.5) -> list[dict]:
    """Build synthetic run result dicts for compute_summary testing."""
    runs = []
    for i in range(n):
        runs.append({
            "seed": i,
            "gene_list": [f"GENE_{i * 10 + j}" for j in range(20)],
            "balanced_accuracy": base_score + i * 0.01,
            "macro_f1": base_score + i * 0.01,
            "per_class_f1": {"0": 0.3, "1": 0.3, "2": 0.3, "3": 0.3},
            "aggregate_score": base_score + i * 0.01,
            "retained_genes": [f"GENE_{i * 10}", f"GENE_{i * 10 + 1}"],
            "n_retained": 2,
        })
    return runs


def test_compute_summary_required_keys():
    """Summary must contain all required keys."""
    runs = _make_runs(5)
    summary = compute_summary(runs)
    required = {
        "n_runs", "balanced_accuracy_mean", "balanced_accuracy_std",
        "balanced_accuracy_min", "balanced_accuracy_max",
        "gene_frequency", "retained_count_distribution",
    }
    assert required.issubset(summary.keys())


def test_compute_summary_n_runs():
    """n_runs must equal the number of runs passed in."""
    runs = _make_runs(8)
    assert compute_summary(runs)["n_runs"] == 8


def test_compute_summary_mean_is_correct():
    """balanced_accuracy_mean must match numpy mean of balanced accuracies."""
    runs = _make_runs(5)
    scores = [r["balanced_accuracy"] for r in runs]
    assert compute_summary(runs)["balanced_accuracy_mean"] == pytest.approx(np.mean(scores))


def test_compute_summary_retained_count_distribution_length():
    """retained_count_distribution must have one entry per run."""
    runs = _make_runs(6)
    summary = compute_summary(runs)
    assert len(summary["retained_count_distribution"]) == 6


def test_compute_summary_gene_frequency_values():
    """gene_frequency counts must be positive integers."""
    runs = _make_runs(5)
    freq = compute_summary(runs)["gene_frequency"]
    assert all(isinstance(v, int) and v > 0 for v in freq.values())


def test_compute_summary_empty_runs():
    """Empty run list must return NaN stats and empty collections."""
    summary = compute_summary([])
    assert summary["n_runs"] == 0
    assert summary["gene_frequency"] == {}
    assert summary["retained_count_distribution"] == []
    import math
    assert math.isnan(summary["balanced_accuracy_mean"])


# ---------------------------------------------------------------------------
# select_best_run tests
# ---------------------------------------------------------------------------


def test_select_best_run_returns_max_aggregate(loader, vocab, evaluator):
    """select_best_run must return the run with the highest aggregate_score."""
    runs = _make_runs(5)
    best = select_best_run(runs)
    assert best["aggregate_score"] == max(r["aggregate_score"] for r in runs)


def test_select_best_run_empty():
    """select_best_run on empty list must return None."""
    assert select_best_run([]) is None
