"""
test_analysis.py — Unit tests for Milestone 8: Gene Retention Tracking & Analysis.

All tests run without a real h5ad file or OpenEvolve API.  Synthetic generation
results are written to a temporary directory to exercise GeneRetentionAnalyzer.

Tests:
  1.  test_load_results_empty_dir          — empty dir → all methods return empty output
  2.  test_compute_retention_frequency_basic — gene A in 2/3 gens → freq ≈ 2/3
  3.  test_retention_frequency_sorted_desc  — higher-count genes rank first
  4.  test_compute_selection_frequency_basic — gene B in 1/3 gens → freq ≈ 1/3
  5.  test_compute_coefficient_stats_basic  — mean coef averaged across gens
  6.  test_coefficient_stats_gene_absent    — genes with 0 coef in all gens → mean=0.0
  7.  test_compare_invocab_vs_outvocab_rates — in/out vocab partition respected
  8.  test_compare_invocab_raises_without_vocab — ValueError if in_vocab not provided
  9.  test_rank_genes_ordering              — higher composite_score ranks first
  10. test_rank_genes_includes_all_seen_genes — every gene across gens appears in output
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import GeneRetentionAnalyzer


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_result(gen_dir: Path, generation_id: int, best_score: float,
                  best_genes: list, retained_genes: list,
                  coefficients: dict | None = None) -> None:
    """Write a synthetic result.json file to gen_dir."""
    gen_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "generation_id": generation_id,
        "best_score": best_score,
        "best_genes": best_genes,
        "retained_genes": retained_genes,
        "coefficients": coefficients or {},
        "all_scores": [best_score],
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    with open(gen_dir / "result.json", "w") as fh:
        json.dump(data, fh)


def _make_3_gens(tmp_path: Path) -> Path:
    """
    Write 3 synthetic generation results:

      Gen 0: selected [A, B, C, D], retained [A, B], coef {A: 0.4, B: 0.2}
      Gen 1: selected [A, B, E, F], retained [A, E], coef {A: 0.6, E: 0.3}
      Gen 2: selected [C, D, E, F], retained [E, F],  coef {E: 0.5, F: 0.1}

    Gene A: retained 2/3 gens (freq=0.667), selected 2/3
    Gene B: retained 1/3 (freq=0.333), selected 2/3
    Gene E: retained 2/3 (freq=0.667), selected 2/3
    Gene C: retained 0/3 (freq=0),     selected 2/3
    """
    results_dir = tmp_path / "results"
    _write_result(results_dir / "gen_0", 0, 0.30,
                  best_genes=["A", "B", "C", "D"],
                  retained_genes=["A", "B"],
                  coefficients={"A": 0.4, "B": 0.2})
    _write_result(results_dir / "gen_1", 1, 0.45,
                  best_genes=["A", "B", "E", "F"],
                  retained_genes=["A", "E"],
                  coefficients={"A": 0.6, "E": 0.3})
    _write_result(results_dir / "gen_2", 2, 0.50,
                  best_genes=["C", "D", "E", "F"],
                  retained_genes=["E", "F"],
                  coefficients={"E": 0.5, "F": 0.1})
    return results_dir


# ── tests ─────────────────────────────────────────────────────────────────────

class TestLoadResults:
    def test_load_results_empty_dir(self, tmp_path):
        """An empty results directory → n_generations=0 and empty DataFrames."""
        results_dir = tmp_path / "empty"
        results_dir.mkdir()
        analyzer = GeneRetentionAnalyzer(str(results_dir))

        assert analyzer.n_generations == 0

        ret_df = analyzer.compute_retention_frequency()
        assert isinstance(ret_df, pd.DataFrame)
        assert len(ret_df) == 0

        sel_df = analyzer.compute_selection_frequency()
        assert len(sel_df) == 0

        coef_df = analyzer.compute_coefficient_stats()
        assert len(coef_df) == 0

        rank_df = analyzer.rank_genes()
        assert len(rank_df) == 0


class TestRetentionFrequency:
    def test_compute_retention_frequency_basic(self, tmp_path):
        """Gene A retained in 2/3 gens → retention_freq ≈ 2/3."""
        results_dir = _make_3_gens(tmp_path)
        analyzer = GeneRetentionAnalyzer(str(results_dir))

        df = analyzer.compute_retention_frequency()
        assert "gene" in df.columns
        assert "retention_count" in df.columns
        assert "retention_freq" in df.columns

        a_row = df[df["gene"] == "A"].iloc[0]
        assert a_row["retention_count"] == 2
        assert abs(a_row["retention_freq"] - 2 / 3) < 1e-9

    def test_retention_frequency_sorted_descending(self, tmp_path):
        """Genes with higher retention_count should appear first."""
        results_dir = _make_3_gens(tmp_path)
        analyzer = GeneRetentionAnalyzer(str(results_dir))

        df = analyzer.compute_retention_frequency()
        counts = df["retention_count"].tolist()
        assert counts == sorted(counts, reverse=True)


class TestSelectionFrequency:
    def test_compute_selection_frequency_basic(self, tmp_path):
        """Gene D selected in 1/3 gens → selection_freq ≈ 1/3."""
        results_dir = _make_3_gens(tmp_path)
        analyzer = GeneRetentionAnalyzer(str(results_dir))

        df = analyzer.compute_selection_frequency()
        assert "gene" in df.columns
        assert "selection_count" in df.columns
        assert "selection_freq" in df.columns

        # D appears in gen_0 and gen_2 → 2/3
        d_row = df[df["gene"] == "D"].iloc[0]
        assert d_row["selection_count"] == 2
        assert abs(d_row["selection_freq"] - 2 / 3) < 1e-9


class TestCoefficientStats:
    def test_compute_coefficient_stats_basic(self, tmp_path):
        """Gene A coef: [0.4, 0.6, 0.0] → mean ≈ 0.333."""
        results_dir = _make_3_gens(tmp_path)
        analyzer = GeneRetentionAnalyzer(str(results_dir))

        df = analyzer.compute_coefficient_stats()
        assert "gene" in df.columns
        assert "mean_coef" in df.columns
        assert "std_coef" in df.columns
        assert "max_coef" in df.columns

        a_row = df[df["gene"] == "A"].iloc[0]
        # Gen0: 0.4, Gen1: 0.6, Gen2: 0.0 → mean = 1.0/3
        assert abs(a_row["mean_coef"] - (0.4 + 0.6 + 0.0) / 3) < 1e-9
        assert abs(a_row["max_coef"] - 0.6) < 1e-9

    def test_coefficient_stats_gene_absent(self, tmp_path):
        """Gene C never in any coef dict → mean_coef = 0.0."""
        results_dir = _make_3_gens(tmp_path)
        analyzer = GeneRetentionAnalyzer(str(results_dir))

        df = analyzer.compute_coefficient_stats()
        c_row = df[df["gene"] == "C"].iloc[0]
        assert c_row["mean_coef"] == 0.0


class TestInVocabComparison:
    def test_compare_invocab_vs_outvocab_rates(self, tmp_path):
        """In-vocab genes (A, B) vs out-of-vocab (C, D, E, F): rates computed correctly."""
        results_dir = _make_3_gens(tmp_path)
        in_vocab = {"A", "B"}
        analyzer = GeneRetentionAnalyzer(str(results_dir), in_vocab=in_vocab)

        result = analyzer.compare_invocab_vs_outvocab()

        assert "in_vocab_retention_rate" in result
        assert "out_vocab_retention_rate" in result
        assert "in_vocab_selection_rate" in result
        assert "out_vocab_selection_rate" in result
        assert "n_in_vocab" in result
        assert "n_out_vocab" in result

        # In-vocab genes (A: retained 2/3, B: retained 1/3) → mean = 0.5
        assert abs(result["in_vocab_retention_rate"] - 0.5) < 1e-9
        assert result["n_in_vocab"] == 2
        assert result["n_out_vocab"] > 0

    def test_compare_invocab_raises_without_vocab(self, tmp_path):
        """ValueError raised if in_vocab was not provided at construction."""
        results_dir = _make_3_gens(tmp_path)
        analyzer = GeneRetentionAnalyzer(str(results_dir))  # no in_vocab

        with pytest.raises(ValueError, match="in_vocab must be provided"):
            analyzer.compare_invocab_vs_outvocab()


class TestRankGenes:
    def test_rank_genes_ordering(self, tmp_path):
        """Genes with higher composite_score (sel_freq × ret_freq) rank first."""
        results_dir = _make_3_gens(tmp_path)
        analyzer = GeneRetentionAnalyzer(str(results_dir))

        df = analyzer.rank_genes()
        assert "gene" in df.columns
        assert "composite_score" in df.columns

        scores = df["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_rank_genes_includes_all_seen_genes(self, tmp_path):
        """Every gene appearing in any generation's best_genes or retained_genes is ranked."""
        results_dir = _make_3_gens(tmp_path)
        analyzer = GeneRetentionAnalyzer(str(results_dir))

        df = analyzer.rank_genes()
        ranked_genes = set(df["gene"].tolist())

        # All genes from our synthetic data: A B C D E F
        expected_genes = {"A", "B", "C", "D", "E", "F"}
        assert expected_genes == ranked_genes

    def test_rank_genes_has_std_coef(self, tmp_path):
        """rank_genes() output must include std_coef consistency column."""
        results_dir = _make_3_gens(tmp_path)
        analyzer = GeneRetentionAnalyzer(str(results_dir))

        df = analyzer.rank_genes()
        assert "std_coef" in df.columns

        # Gene A coef: [0.4, 0.6, 0.0] → std > 0
        a_row = df[df["gene"] == "A"].iloc[0]
        assert a_row["std_coef"] > 0.0

        # Gene C never in coef dict → std_coef == 0.0
        c_row = df[df["gene"] == "C"].iloc[0]
        assert c_row["std_coef"] == 0.0


class TestCompareEvolutionVsBaseline:
    def test_compare_evolution_vs_baseline_basic(self, tmp_path):
        """High evolution scores vs low baseline → evolution_mean > baseline_mean and valid result."""
        results_dir = _make_3_gens(tmp_path)
        # Our synthetic gens have scores [0.30, 0.45, 0.50]
        analyzer = GeneRetentionAnalyzer(str(results_dir))

        baseline_scores = [0.10, 0.12, 0.11, 0.13, 0.09]
        result = analyzer.compare_evolution_vs_baseline(baseline_scores)

        assert "evolution_mean" in result
        assert "baseline_mean" in result
        assert "p_value" in result
        assert "effect_size_rank_biserial" in result
        assert "n_evolution" in result
        assert "n_baseline" in result
        assert "statistic" in result
        assert "evolution_ci_95" in result
        assert "baseline_ci_95" in result

        # Evolution (mean ≈ 0.42) clearly above baseline (mean ≈ 0.11)
        assert result["evolution_mean"] > result["baseline_mean"]
        assert result["n_evolution"] == 3
        assert result["n_baseline"] == 5
        # Effect size should be positive (evolution > baseline)
        assert result["effect_size_rank_biserial"] > 0

    def test_compare_evolution_vs_baseline_empty_evolution(self, tmp_path):
        """Empty results dir → returns dict with NaN values, no exception."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        analyzer = GeneRetentionAnalyzer(str(empty_dir))

        result = analyzer.compare_evolution_vs_baseline([0.25, 0.26, 0.24])

        import math
        assert math.isnan(result["evolution_mean"])
        assert math.isnan(result["p_value"])
        assert result["n_evolution"] == 0

    def test_compare_evolution_vs_baseline_empty_baseline(self, tmp_path):
        """Empty baseline list → returns dict with NaN values, no exception."""
        results_dir = _make_3_gens(tmp_path)
        analyzer = GeneRetentionAnalyzer(str(results_dir))

        result = analyzer.compare_evolution_vs_baseline([])

        import math
        assert math.isnan(result["p_value"])
        assert result["n_baseline"] == 0
