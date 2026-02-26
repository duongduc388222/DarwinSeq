"""
tests/test_evaluator.py — Unit tests for LASSOEvaluator and EvalResult.

All tests use synthetic DataFrames — no real h5ad file is required.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluator import EvalResult, LASSOEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluator(tmp_path, alpha=0.01):
    """Write a minimal lasso_params.json and return a LASSOEvaluator."""
    import json
    cfg = {
        "_notes": "test config",
        "model_type": "Lasso",
        "fit_one_per_target": True,
        "alpha": alpha,
        "fit_intercept": True,
        "max_iter": 10000,
        "random_state": 42,
        "scaler": "StandardScaler",
        "scoring_metric": "pearson_r",
        "aggregate_fn": "mean",
        "nan_policy": "drop_per_target",
        "coef_threshold": 1e-10,
    }
    cfg_path = tmp_path / "lasso_params.json"
    cfg_path.write_text(json.dumps(cfg))
    return LASSOEvaluator(config_path=str(cfg_path))


def _make_sparse_data(n=80, n_genes=20, signal_genes=None, seed=0):
    """
    Build a synthetic (X, y) pair where y is a linear combination of a small
    set of signal_genes plus Gaussian noise.

    Args:
        n: Number of samples.
        n_genes: Total number of genes.
        signal_genes: List of column indices carrying the signal (default [0,1]).
        seed: Random seed.

    Returns:
        (X, y) as DataFrames with gene_0..gene_{n_genes-1} columns and
        target_0 as the single target column.
    """
    if signal_genes is None:
        signal_genes = [0, 1]
    rng = np.random.default_rng(seed)
    X_vals = rng.standard_normal((n, n_genes))
    coefs = np.zeros(n_genes)
    for idx in signal_genes:
        coefs[idx] = rng.uniform(1.0, 3.0)
    y_vals = X_vals @ coefs + 0.05 * rng.standard_normal(n)

    gene_cols = [f"gene_{i}" for i in range(n_genes)]
    X = pd.DataFrame(X_vals, columns=gene_cols)
    y = pd.DataFrame({"target_0": y_vals})
    return X, y, [f"gene_{i}" for i in signal_genes]


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------

def test_missing_config_raises(tmp_path):
    """FileNotFoundError raised when config path does not exist."""
    with pytest.raises(FileNotFoundError):
        LASSOEvaluator(config_path=str(tmp_path / "nonexistent.json"))


def test_default_config_loads():
    """Default config file (config/lasso_params.json) loads without error."""
    ev = LASSOEvaluator()
    assert ev._alpha > 0


# ---------------------------------------------------------------------------
# 2. Shape mismatch
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises(tmp_path):
    """ValueError raised when X and y have different numbers of rows."""
    ev = _make_evaluator(tmp_path)
    X = pd.DataFrame(np.ones((10, 5)), columns=[f"g{i}" for i in range(5)])
    y = pd.DataFrame({"t": np.ones(8)})
    with pytest.raises(ValueError, match="same number of rows"):
        ev.evaluate(X, y)


# ---------------------------------------------------------------------------
# 3. Empty inputs
# ---------------------------------------------------------------------------

def test_empty_X_returns_default(tmp_path):
    """Empty X returns a zero-score EvalResult without crashing."""
    ev = _make_evaluator(tmp_path)
    X = pd.DataFrame(columns=["g0", "g1"])
    y = pd.DataFrame(columns=["t0"])
    result = ev.evaluate(X, y)
    assert result.aggregate_score == 0.0
    assert result.retained_genes == []


# ---------------------------------------------------------------------------
# 4. Synthetic signal recovery
# ---------------------------------------------------------------------------

def test_signal_genes_retained(tmp_path):
    """
    With a small alpha, LASSO should assign non-zero coefficients to genes
    that carry the true signal.
    """
    ev = _make_evaluator(tmp_path, alpha=0.001)
    X, y, signal_gene_names = _make_sparse_data(n=100, n_genes=20, signal_genes=[0, 1])
    result = ev.evaluate(X, y)
    for sg in signal_gene_names:
        assert sg in result.retained_genes, (
            f"Expected signal gene '{sg}' in retained_genes, got {result.retained_genes}"
        )


# ---------------------------------------------------------------------------
# 5. Retained genes subset property
# ---------------------------------------------------------------------------

def test_retained_genes_subset_of_input_columns(tmp_path):
    """retained_genes must be a subset of input gene column names."""
    ev = _make_evaluator(tmp_path)
    X, y, _ = _make_sparse_data(n=80, n_genes=15)
    result = ev.evaluate(X, y)
    assert set(result.retained_genes) <= set(X.columns)


# ---------------------------------------------------------------------------
# 6. Score range
# ---------------------------------------------------------------------------

def test_scores_in_valid_range(tmp_path):
    """All per-target Pearson r values must lie in [-1, 1]."""
    ev = _make_evaluator(tmp_path)
    X, y, _ = _make_sparse_data(n=80, n_genes=15)
    result = ev.evaluate(X, y)
    for target, r in result.scores.items():
        assert -1.0 <= r <= 1.0, f"Score for '{target}' out of range: {r}"
    assert -1.0 <= result.aggregate_score <= 1.0


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------

def test_determinism(tmp_path):
    """Same X and y must produce identical EvalResult on two consecutive calls."""
    ev = _make_evaluator(tmp_path)
    X, y, _ = _make_sparse_data(n=80, n_genes=15)
    r1 = ev.evaluate(X, y)
    r2 = ev.evaluate(X, y)
    assert r1.aggregate_score == r2.aggregate_score
    assert r1.retained_genes == r2.retained_genes
    assert r1.scores == r2.scores
    assert r1.coefficients == r2.coefficients


# ---------------------------------------------------------------------------
# 8. NaN handling in y
# ---------------------------------------------------------------------------

def test_all_nan_target_skipped(tmp_path):
    """A target column that is entirely NaN is skipped gracefully."""
    ev = _make_evaluator(tmp_path)
    X, y, _ = _make_sparse_data(n=80, n_genes=10)
    y["all_nan_target"] = float("nan")
    result = ev.evaluate(X, y)
    assert "all_nan_target" not in result.scores


def test_partial_nan_target_uses_valid_rows(tmp_path):
    """A target with some NaN values is still evaluated on non-NaN rows."""
    ev = _make_evaluator(tmp_path, alpha=0.001)
    X, y, _ = _make_sparse_data(n=80, n_genes=10)
    # Zero out half the target values to NaN.
    y_partial = y.copy()
    y_partial.loc[y_partial.index[:40], "target_0"] = float("nan")
    result = ev.evaluate(X, y_partial)
    assert "target_0" in result.scores


# ---------------------------------------------------------------------------
# 9. Multi-target: union of retained genes
# ---------------------------------------------------------------------------

def test_multi_target_union_retained(tmp_path):
    """
    With two targets driven by disjoint signal genes, retained_genes should
    cover genes from both targets.
    """
    ev = _make_evaluator(tmp_path, alpha=0.001)
    rng = np.random.default_rng(99)
    n, n_genes = 100, 20
    X_vals = rng.standard_normal((n, n_genes))
    gene_cols = [f"gene_{i}" for i in range(n_genes)]

    # target_0 driven by gene_0; target_1 driven by gene_10.
    y0 = X_vals[:, 0] * 3.0 + 0.05 * rng.standard_normal(n)
    y1 = X_vals[:, 10] * 3.0 + 0.05 * rng.standard_normal(n)
    X = pd.DataFrame(X_vals, columns=gene_cols)
    y = pd.DataFrame({"t0": y0, "t1": y1})

    result = ev.evaluate(X, y)
    assert "gene_0" in result.retained_genes
    assert "gene_10" in result.retained_genes
    assert len(result.scores) == 2


# ---------------------------------------------------------------------------
# 10. n_retained consistency
# ---------------------------------------------------------------------------

def test_n_retained_matches_retained_genes_length(tmp_path):
    """EvalResult.n_retained must equal len(retained_genes)."""
    ev = _make_evaluator(tmp_path)
    X, y, _ = _make_sparse_data(n=80, n_genes=15)
    result = ev.evaluate(X, y)
    assert result.n_retained == len(result.retained_genes)


# ---------------------------------------------------------------------------
# 11. Coefficients keys match retained_genes
# ---------------------------------------------------------------------------

def test_coefficients_keys_match_retained_genes(tmp_path):
    """coefficients dict keys must equal the set of retained_genes."""
    ev = _make_evaluator(tmp_path)
    X, y, _ = _make_sparse_data(n=80, n_genes=15)
    result = ev.evaluate(X, y)
    assert set(result.coefficients.keys()) == set(result.retained_genes)


# ---------------------------------------------------------------------------
# 12. EvalResult is a plain dataclass (picklable / inspectable)
# ---------------------------------------------------------------------------

def test_evalresult_fields():
    """EvalResult can be constructed directly and fields are accessible."""
    r = EvalResult(
        scores={"t0": 0.5},
        aggregate_score=0.5,
        retained_genes=["g1"],
        coefficients={"g1": 0.3},
        n_retained=1,
    )
    assert r.scores == {"t0": 0.5}
    assert r.aggregate_score == 0.5
    assert r.retained_genes == ["g1"]
    assert r.n_retained == 1
