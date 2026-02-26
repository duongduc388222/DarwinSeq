"""
tests/test_evaluator.py — Unit tests for ADNCEvaluator and EvalResult.

All tests use synthetic DataFrames — no real h5ad file is required.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluator import ADNCEvaluator, EvalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evaluator(tmp_path):
    """
    Write a minimal model_params.json and return an ADNCEvaluator.

    Args:
        tmp_path: pytest-provided temporary directory (pathlib.Path).

    Returns:
        ADNCEvaluator instance configured with small max_iter for fast tests.
    """
    cfg = {
        "model_type": "LogisticRegression",
        "target_col": "ADNC",
        "l1_ratio": 1,
        "solver": "liblinear",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": 42,
        "scaler": "StandardScaler",
        "scoring_metric": "balanced_accuracy",
        "secondary_metric": "macro_f1",
        "coef_threshold": 1e-10,
    }
    cfg_path = tmp_path / "model_params.json"
    cfg_path.write_text(json.dumps(cfg))
    return ADNCEvaluator(config_path=str(cfg_path))


def _make_4class_data(n_per_class=30, n_genes=50, seed=0):
    """
    Build a synthetic (X, y) pair with 4 balanced ADNC classes.

    Each class is driven by a disjoint set of signal genes to ensure
    the classifier can achieve above-chance balanced accuracy.

    Args:
        n_per_class: Number of samples per ADNC class.
        n_genes: Total number of genes (features).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
          - X: DataFrame of shape (n_per_class * 4, n_genes).
          - y: Single-column DataFrame with column "ADNC" containing integers 0–3.
    """
    rng = np.random.default_rng(seed)
    n_total = n_per_class * 4
    X_vals = rng.standard_normal((n_total, n_genes))
    y_vals = np.repeat([0, 1, 2, 3], n_per_class)

    # Add class-specific signal: each class boosts a disjoint block of genes.
    block = n_genes // 4
    for cls in range(4):
        mask = y_vals == cls
        X_vals[mask, cls * block:(cls + 1) * block] += 3.0

    gene_cols = [f"gene_{i}" for i in range(n_genes)]
    X = pd.DataFrame(X_vals, columns=gene_cols)
    y = pd.DataFrame({"ADNC": y_vals.astype(float)})
    return X, y


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------


def test_missing_config_raises(tmp_path):
    """FileNotFoundError raised when config path does not exist."""
    with pytest.raises(FileNotFoundError):
        ADNCEvaluator(config_path=str(tmp_path / "nonexistent.json"))


def test_default_config_loads():
    """Default config file (config/model_params.json) loads without error."""
    ev = ADNCEvaluator()
    assert ev._C > 0
    assert ev._l1_ratio == 1.0


# ---------------------------------------------------------------------------
# 2. Shape mismatch
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises(tmp_path):
    """ValueError raised when X and y have different numbers of rows."""
    ev = _make_evaluator(tmp_path)
    X = pd.DataFrame(np.ones((10, 5)), columns=[f"g{i}" for i in range(5)])
    y = pd.DataFrame({"ADNC": np.ones(8)})
    with pytest.raises(ValueError, match="same number of rows"):
        ev.evaluate(X, y)


# ---------------------------------------------------------------------------
# 3. Empty inputs
# ---------------------------------------------------------------------------


def test_empty_returns_default(tmp_path):
    """Empty X returns a zero-score EvalResult without crashing."""
    ev = _make_evaluator(tmp_path)
    X = pd.DataFrame(columns=["g0", "g1"])
    y = pd.DataFrame(columns=["ADNC"])
    result = ev.evaluate(X, y)
    assert result.aggregate_score == 0.0
    assert result.balanced_accuracy == 0.0
    assert result.retained_genes == []


# ---------------------------------------------------------------------------
# 4. Basic 4-class classification
# ---------------------------------------------------------------------------


def test_basic_4class(tmp_path):
    """
    With 4 balanced classes and class-specific signal genes, balanced
    accuracy and macro F1 must be above chance (> 0.25 for 4 classes).
    """
    ev = _make_evaluator(tmp_path)
    X, y = _make_4class_data(n_per_class=30, n_genes=50)
    result = ev.evaluate(X, y)
    assert result.balanced_accuracy > 0.25, (
        f"Expected balanced_accuracy > 0.25 but got {result.balanced_accuracy:.4f}"
    )
    assert result.macro_f1 > 0.0


# ---------------------------------------------------------------------------
# 5. Degenerate: single class present
# ---------------------------------------------------------------------------


def test_degenerate_single_class(tmp_path):
    """All cells with the same ADNC class returns zero result without error."""
    ev = _make_evaluator(tmp_path)
    n = 40
    X = pd.DataFrame(np.random.randn(n, 10), columns=[f"g{i}" for i in range(10)])
    y = pd.DataFrame({"ADNC": np.zeros(n)})  # only class 0
    result = ev.evaluate(X, y)
    assert result.balanced_accuracy == 0.0
    assert result.macro_f1 == 0.0
    assert result.retained_genes == []


# ---------------------------------------------------------------------------
# 6. All NaN returns zero result
# ---------------------------------------------------------------------------


def test_all_nan_returns_zero(tmp_path):
    """All-NaN ADNC values return zero result without crashing."""
    ev = _make_evaluator(tmp_path)
    n = 40
    X = pd.DataFrame(np.random.randn(n, 10), columns=[f"g{i}" for i in range(10)])
    y = pd.DataFrame({"ADNC": [float("nan")] * n})
    result = ev.evaluate(X, y)
    assert result.balanced_accuracy == 0.0
    assert result.retained_genes == []


# ---------------------------------------------------------------------------
# 7. NaN rows dropped before fitting
# ---------------------------------------------------------------------------


def test_nan_rows_dropped(tmp_path):
    """
    NaN rows in y are silently dropped; the evaluator still runs on valid rows.
    With ≥ 2 classes in the non-NaN subset it must not crash and must return
    a non-trivially zero result.
    """
    ev = _make_evaluator(tmp_path)
    X, y = _make_4class_data(n_per_class=30, n_genes=50)
    # Zero out half the labels.
    y_partial = y.copy()
    y_partial.iloc[: len(y) // 2, 0] = float("nan")
    # Should not raise; should still produce a result.
    result = ev.evaluate(X, y_partial)
    assert isinstance(result.balanced_accuracy, float)


# ---------------------------------------------------------------------------
# 8. Retained genes are a subset of input columns
# ---------------------------------------------------------------------------


def test_retained_genes_subset(tmp_path):
    """retained_genes must be a subset of the gene column names in X."""
    ev = _make_evaluator(tmp_path)
    X, y = _make_4class_data(n_per_class=25, n_genes=40)
    result = ev.evaluate(X, y)
    assert set(result.retained_genes) <= set(X.columns), (
        f"retained_genes contains names not in X.columns: "
        f"{set(result.retained_genes) - set(X.columns)}"
    )


# ---------------------------------------------------------------------------
# 9. Determinism
# ---------------------------------------------------------------------------


def test_deterministic(tmp_path):
    """Same (X, y) called twice must produce identical EvalResult."""
    ev = _make_evaluator(tmp_path)
    X, y = _make_4class_data(n_per_class=25, n_genes=40)
    r1 = ev.evaluate(X, y)
    r2 = ev.evaluate(X, y)
    assert r1.balanced_accuracy == r2.balanced_accuracy
    assert r1.macro_f1 == r2.macro_f1
    assert r1.retained_genes == r2.retained_genes
    assert r1.coefficients == r2.coefficients
    assert r1.per_class_f1 == r2.per_class_f1


# ---------------------------------------------------------------------------
# 10. aggregate_score is an alias for balanced_accuracy
# ---------------------------------------------------------------------------


def test_aggregate_score_alias(tmp_path):
    """EvalResult.aggregate_score must equal EvalResult.balanced_accuracy."""
    ev = _make_evaluator(tmp_path)
    X, y = _make_4class_data(n_per_class=25, n_genes=40)
    result = ev.evaluate(X, y)
    assert result.aggregate_score == result.balanced_accuracy


# ---------------------------------------------------------------------------
# 11. n_retained consistency
# ---------------------------------------------------------------------------


def test_n_retained_consistency(tmp_path):
    """EvalResult.n_retained must equal len(retained_genes)."""
    ev = _make_evaluator(tmp_path)
    X, y = _make_4class_data(n_per_class=25, n_genes=40)
    result = ev.evaluate(X, y)
    assert result.n_retained == len(result.retained_genes)


# ---------------------------------------------------------------------------
# 12. Coefficients keys match retained_genes
# ---------------------------------------------------------------------------


def test_coefficients_keys_match_retained(tmp_path):
    """coefficients dict keys must equal the set of retained_genes."""
    ev = _make_evaluator(tmp_path)
    X, y = _make_4class_data(n_per_class=25, n_genes=40)
    result = ev.evaluate(X, y)
    assert set(result.coefficients.keys()) == set(result.retained_genes)


# ---------------------------------------------------------------------------
# 13. per_class_f1 covers present classes
# ---------------------------------------------------------------------------


def test_per_class_f1_keys(tmp_path):
    """per_class_f1 must contain string keys for each class in the data."""
    ev = _make_evaluator(tmp_path)
    X, y = _make_4class_data(n_per_class=25, n_genes=40)
    result = ev.evaluate(X, y)
    # All 4 classes are present in the synthetic data.
    assert set(result.per_class_f1.keys()) == {"0", "1", "2", "3"}
    for cls_f1 in result.per_class_f1.values():
        assert 0.0 <= cls_f1 <= 1.0


# ---------------------------------------------------------------------------
# 14. EvalResult is a plain dataclass (constructable directly)
# ---------------------------------------------------------------------------


def test_evalresult_fields():
    """EvalResult can be constructed directly and fields are accessible."""
    r = EvalResult(
        balanced_accuracy=0.6,
        macro_f1=0.55,
        aggregate_score=0.6,
        retained_genes=["APOE", "TREM2"],
        coefficients={"APOE": 0.4, "TREM2": 0.3},
        n_retained=2,
        per_class_f1={"0": 0.7, "1": 0.5, "2": 0.6, "3": 0.65},
    )
    assert r.balanced_accuracy == 0.6
    assert r.aggregate_score == r.balanced_accuracy
    assert r.n_retained == 2
    assert "APOE" in r.retained_genes
    assert r.per_class_f1["0"] == 0.7
