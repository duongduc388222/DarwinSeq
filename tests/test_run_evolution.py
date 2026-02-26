"""
test_run_evolution.py — Unit tests for scripts/run_evolution.py and
the generate_summary() function added to src/evolve.py (Milestone 7).

All tests use mocks/stubs so no real h5ad file or API calls are needed.
"""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# Allow importing from project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_evolution import _ensure_openevolve, _print_summary, main, parse_args
from src.evolve import GenerationResult, generate_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results(n: int = 3) -> list[GenerationResult]:
    """Build synthetic GenerationResult list for testing."""
    results = []
    for i in range(n):
        results.append(
            GenerationResult(
                generation_id=i,
                best_score=0.10 + i * 0.05,
                best_genes=[f"GENE_{j}" for j in range(200)],
                retained_genes=[f"GENE_{j}" for j in range(5)],
                coefficients={f"GENE_{j}": float(5 - j) * 0.1 for j in range(5)},
                all_scores=[0.10 + i * 0.05],
            )
        )
    return results


# ---------------------------------------------------------------------------
# 1. parse_args — default values
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    """parse_args() with no arguments returns expected defaults."""
    args = parse_args([])
    assert args.config == "config/evolve_config.yaml"
    assert args.n_generations == 5
    assert args.output_dir == "results/evolution"
    assert args.baseline_score is None


# ---------------------------------------------------------------------------
# 2. parse_args — custom overrides
# ---------------------------------------------------------------------------


def test_parse_args_overrides():
    """parse_args() correctly stores custom argument values."""
    args = parse_args([
        "--config", "my_config.yaml",
        "--n_generations", "10",
        "--output_dir", "/tmp/evo",
        "--baseline_score", "0.15",
    ])
    assert args.config == "my_config.yaml"
    assert args.n_generations == 10
    assert args.output_dir == "/tmp/evo"
    assert args.baseline_score == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# 3. main() exits when openevolve is not installed
# ---------------------------------------------------------------------------


def test_main_no_openevolve(tmp_path):
    """main() exits with code 1 when _ensure_openevolve raises SystemExit."""
    # Patch _ensure_openevolve to simulate openevolve not being installed.
    with mock.patch("scripts.run_evolution._ensure_openevolve", side_effect=SystemExit(1)):
        with pytest.raises(SystemExit) as exc_info:
            main(["--output_dir", str(tmp_path)])
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# 4. main() writes summary.json with correct content
# ---------------------------------------------------------------------------


def test_main_writes_summary_json(tmp_path):
    """main() writes summary.json to output_dir after mocked evolution run."""
    fake_results = _make_results(3)

    with mock.patch("scripts.run_evolution._ensure_openevolve"), \
         mock.patch("scripts.run_evolution.EvolutionRunner") as mock_runner_cls:
        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = fake_results

        main([
            "--output_dir", str(tmp_path),
            "--n_generations", "3",
            "--baseline_score", "0.12",
        ])

    summary_path = tmp_path / "summary.json"
    assert summary_path.exists(), "summary.json must be created"

    summary = json.loads(summary_path.read_text())
    assert summary["best_score"] == pytest.approx(0.20)
    assert summary["best_generation"] == 2
    assert summary["baseline_mean_score"] == pytest.approx(0.12)
    assert len(summary["generations"]) == 3
    assert "elapsed_seconds" in summary


# ---------------------------------------------------------------------------
# 5. generate_summary() — unit tests for src/evolve.py
# ---------------------------------------------------------------------------


def test_generate_summary_basic():
    """generate_summary picks the best generation and structures output correctly."""
    results = [
        GenerationResult(
            generation_id=0,
            best_score=0.10,
            retained_genes=["A", "B"],
            coefficients={"A": 0.8, "B": 0.2},
        ),
        GenerationResult(
            generation_id=1,
            best_score=0.25,
            retained_genes=["A", "C"],
            coefficients={"A": 0.9, "C": 0.1},
        ),
    ]
    summary = generate_summary(results, baseline_mean_score=0.12)

    assert summary["best_score"] == pytest.approx(0.25)
    assert summary["best_generation"] == 1
    assert summary["baseline_mean_score"] == pytest.approx(0.12)
    assert len(summary["generations"]) == 2

    # Top genes should be sorted by coefficient descending.
    gen0 = summary["generations"][0]
    assert gen0["top_genes"][0] == "A"  # A has higher coef than B
    assert gen0["n_retained"] == 2


def test_generate_summary_empty():
    """generate_summary returns safe empty structure for empty results list."""
    summary = generate_summary([], baseline_mean_score=0.15)
    assert summary["best_score"] is None
    assert summary["best_generation"] is None
    assert summary["generations"] == []
    assert summary["baseline_mean_score"] == pytest.approx(0.15)
