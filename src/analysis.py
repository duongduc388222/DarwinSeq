"""
analysis.py — Gene retention tracking and ranking for DarwinSeq evolution results.

Loads per-generation result JSON files produced by EvolutionRunner.log_generation()
and computes statistics about gene selection and LASSO retention patterns across
all generations.

Usage example:
    analyzer = GeneRetentionAnalyzer("results/evolution", in_vocab=my_in_vocab_set)
    retention_df = analyzer.compute_retention_frequency()
    rankings = analyzer.rank_genes()
    rankings.to_csv("results/analysis/gene_rankings.csv", index=False)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GeneRetentionAnalyzer:
    """
    Analyze gene retention and selection patterns across OpenEvolve generations.

    Loads all gen_*/result.json files from results_dir and provides methods to
    compute per-gene statistics over all generations.

    Args:
        results_dir: Path to the evolution output directory containing gen_* subdirs.
        in_vocab: Optional set of in-vocabulary gene symbols. Required only for
                  compare_invocab_vs_outvocab(). Can be passed at construction or
                  later via that method's in_vocab parameter.
    """

    def __init__(self, results_dir: str, in_vocab: set | None = None) -> None:
        self._results_dir = Path(results_dir)
        self._in_vocab: set | None = in_vocab
        self._records: list[dict] = self._load_records()

        if not self._records:
            logger.warning(
                "No generation results found in '%s'. All methods will return empty output.",
                results_dir,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def compute_retention_frequency(self) -> pd.DataFrame:
        """
        Count how often each gene is retained by LASSO (non-zero coefficient) across
        all loaded generations.

        Returns:
            DataFrame with columns:
              - gene (str): Gene symbol.
              - retention_count (int): Number of generations in which LASSO retained it.
              - retention_freq (float): retention_count / n_generations (0–1).
            Sorted by retention_count descending. Empty DataFrame if no results loaded.
        """
        if not self._records:
            return pd.DataFrame(columns=["gene", "retention_count", "retention_freq"])

        n_gens = len(self._records)
        counts: dict[str, int] = {}
        for rec in self._records:
            for gene in rec.get("retained_genes", []):
                counts[gene] = counts.get(gene, 0) + 1

        rows = [
            {"gene": g, "retention_count": c, "retention_freq": c / n_gens}
            for g, c in counts.items()
        ]
        df = pd.DataFrame(rows, columns=["gene", "retention_count", "retention_freq"])
        return df.sort_values("retention_count", ascending=False).reset_index(drop=True)

    def compute_selection_frequency(self) -> pd.DataFrame:
        """
        Count how often each gene is selected by the LLM across all loaded generations.

        A gene is "selected" if it appears in that generation's 200-gene list
        (best_genes in result.json).

        Returns:
            DataFrame with columns:
              - gene (str): Gene symbol.
              - selection_count (int): Number of generations in which it was selected.
              - selection_freq (float): selection_count / n_generations (0–1).
            Sorted by selection_count descending. Empty DataFrame if no results loaded.
        """
        if not self._records:
            return pd.DataFrame(columns=["gene", "selection_count", "selection_freq"])

        n_gens = len(self._records)
        counts: dict[str, int] = {}
        for rec in self._records:
            for gene in rec.get("best_genes", []):
                counts[gene] = counts.get(gene, 0) + 1

        rows = [
            {"gene": g, "selection_count": c, "selection_freq": c / n_gens}
            for g, c in counts.items()
        ]
        df = pd.DataFrame(rows, columns=["gene", "selection_count", "selection_freq"])
        return df.sort_values("selection_count", ascending=False).reset_index(drop=True)

    def compute_coefficient_stats(self) -> pd.DataFrame:
        """
        Compute mean, standard deviation, and maximum of the absolute LASSO coefficient
        for each gene across all generations.

        Genes that are not retained in a generation contribute 0.0 to that generation's
        coefficient value (they were zeroed out by the LASSO penalty).

        Returns:
            DataFrame with columns:
              - gene (str): Gene symbol.
              - mean_coef (float): Mean |coefficient| across all generations.
              - std_coef (float): Std dev of |coefficient| across all generations.
              - max_coef (float): Maximum |coefficient| observed across all generations.
            Sorted by mean_coef descending. Empty DataFrame if no results loaded.
        """
        if not self._records:
            return pd.DataFrame(columns=["gene", "mean_coef", "std_coef", "max_coef"])

        n_gens = len(self._records)

        # Collect all genes seen across any generation.
        all_genes: set[str] = set()
        for rec in self._records:
            all_genes.update(rec.get("coefficients", {}).keys())
            all_genes.update(rec.get("best_genes", []))

        rows = []
        for gene in all_genes:
            values = []
            for rec in self._records:
                coef = rec.get("coefficients", {}).get(gene, 0.0)
                values.append(float(coef))
            rows.append(
                {
                    "gene": gene,
                    "mean_coef": float(np.mean(values)),
                    "std_coef": float(np.std(values)),
                    "max_coef": float(np.max(values)),
                }
            )

        df = pd.DataFrame(rows, columns=["gene", "mean_coef", "std_coef", "max_coef"])
        return df.sort_values("mean_coef", ascending=False).reset_index(drop=True)

    def compare_invocab_vs_outvocab(self) -> dict:
        """
        Compare LASSO retention and LLM selection rates for in-vocabulary vs
        out-of-vocabulary genes.

        Uses the in_vocab set provided at construction time.

        Returns:
            Dict with keys:
              - in_vocab_retention_rate (float): Fraction of in-vocab genes retained (mean).
              - out_vocab_retention_rate (float): Fraction of out-vocab genes retained (mean).
              - in_vocab_selection_rate (float): Fraction of in-vocab genes selected (mean).
              - out_vocab_selection_rate (float): Fraction of out-vocab genes selected (mean).
              - n_in_vocab (int): Number of in-vocab genes seen across all generations.
              - n_out_vocab (int): Number of out-vocab genes seen across all generations.

        Raises:
            ValueError: If in_vocab was not provided at construction.
        """
        if self._in_vocab is None:
            raise ValueError(
                "in_vocab must be provided at GeneRetentionAnalyzer() construction "
                "to use compare_invocab_vs_outvocab()."
            )

        if not self._records:
            return {
                "in_vocab_retention_rate": float("nan"),
                "out_vocab_retention_rate": float("nan"),
                "in_vocab_selection_rate": float("nan"),
                "out_vocab_selection_rate": float("nan"),
                "n_in_vocab": 0,
                "n_out_vocab": 0,
            }

        retention_df = self.compute_retention_frequency()
        selection_df = self.compute_selection_frequency()

        ret_map = dict(zip(retention_df["gene"], retention_df["retention_freq"]))
        sel_map = dict(zip(selection_df["gene"], selection_df["selection_freq"]))

        all_genes: set[str] = set()
        for rec in self._records:
            all_genes.update(rec.get("best_genes", []))
            all_genes.update(rec.get("retained_genes", []))

        in_genes = [g for g in all_genes if g in self._in_vocab]
        out_genes = [g for g in all_genes if g not in self._in_vocab]

        def _mean_freq(genes: list[str], freq_map: dict[str, float]) -> float:
            if not genes:
                return float("nan")
            return float(np.mean([freq_map.get(g, 0.0) for g in genes]))

        return {
            "in_vocab_retention_rate": _mean_freq(in_genes, ret_map),
            "out_vocab_retention_rate": _mean_freq(out_genes, ret_map),
            "in_vocab_selection_rate": _mean_freq(in_genes, sel_map),
            "out_vocab_selection_rate": _mean_freq(out_genes, sel_map),
            "n_in_vocab": len(in_genes),
            "n_out_vocab": len(out_genes),
        }

    def rank_genes(self) -> pd.DataFrame:
        """
        Rank genes by a composite score that rewards both frequent LLM selection
        and frequent LASSO-for-classification retention.

        composite_score = selection_freq × retention_freq

        Also includes mean_coef (higher = stronger ADNC discriminator on average)
        and std_coef (lower = more consistent signal across generations).

        Returns:
            DataFrame with columns:
              - gene (str): Gene symbol.
              - selection_freq (float): Fraction of generations in which LLM selected it.
              - retention_freq (float): Fraction of generations in which LASSO retained it.
              - composite_score (float): selection_freq × retention_freq.
              - mean_coef (float): Mean sum-of-|coef| across all generations.
              - std_coef (float): Std dev of sum-of-|coef|; lower = more consistent.
            Sorted by composite_score descending (ties broken by mean_coef).
            Empty DataFrame if no results loaded.
        """
        if not self._records:
            return pd.DataFrame(
                columns=[
                    "gene", "selection_freq", "retention_freq",
                    "composite_score", "mean_coef", "std_coef",
                ]
            )

        sel_df = self.compute_selection_frequency()
        ret_df = self.compute_retention_frequency()
        coef_df = self.compute_coefficient_stats()

        sel_map = dict(zip(sel_df["gene"], sel_df["selection_freq"]))
        ret_map = dict(zip(ret_df["gene"], ret_df["retention_freq"]))
        coef_mean_map = dict(zip(coef_df["gene"], coef_df["mean_coef"]))
        coef_std_map = dict(zip(coef_df["gene"], coef_df["std_coef"]))

        # Union of all genes seen in any generation.
        all_genes: set[str] = set()
        for rec in self._records:
            all_genes.update(rec.get("best_genes", []))
            all_genes.update(rec.get("retained_genes", []))

        rows = []
        for gene in all_genes:
            s = sel_map.get(gene, 0.0)
            r = ret_map.get(gene, 0.0)
            rows.append(
                {
                    "gene": gene,
                    "selection_freq": s,
                    "retention_freq": r,
                    "composite_score": s * r,
                    "mean_coef": coef_mean_map.get(gene, 0.0),
                    "std_coef": coef_std_map.get(gene, 0.0),
                }
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "gene", "selection_freq", "retention_freq",
                "composite_score", "mean_coef", "std_coef",
            ],
        )
        return (
            df.sort_values(["composite_score", "mean_coef"], ascending=[False, False])
            .reset_index(drop=True)
        )

    def compare_evolution_vs_baseline(self, baseline_scores: list[float]) -> dict:
        """
        Compare evolution generation scores against a random-baseline distribution
        using a one-sided Mann-Whitney U test (H1: evolution > baseline).

        Both evolution and baseline scores are balanced accuracy values (0–1) from
        ADNC classification. Uses scipy.stats.mannwhitneyu which is included in the
        project's dependencies.

        Args:
            baseline_scores: List of balanced-accuracy scores from the random-baseline
                             run (e.g. from results/baseline/all_runs.json).

        Returns:
            Dict with keys:
              - evolution_mean (float): Mean balanced accuracy of evolution generations.
              - baseline_mean (float): Mean balanced accuracy of baseline runs.
              - n_evolution (int): Number of evolution generations.
              - n_baseline (int): Number of baseline runs.
              - statistic (float): Mann-Whitney U statistic.
              - p_value (float): One-sided p-value (evolution > baseline).
              - effect_size_rank_biserial (float): Rank-biserial correlation (0–1 range;
                  positive = evolution tends to be higher).
              - evolution_ci_95 (tuple[float, float]): 2.5th and 97.5th percentile of
                  evolution scores (NaN, NaN if fewer than 3 generations).
              - baseline_ci_95 (tuple[float, float]): Same for baseline scores.
            All numeric fields are NaN if either list is empty.
        """
        from scipy import stats

        nan_result = {
            "evolution_mean": float("nan"),
            "baseline_mean": float("nan"),
            "n_evolution": len(self._records),
            "n_baseline": len(baseline_scores),
            "statistic": float("nan"),
            "p_value": float("nan"),
            "effect_size_rank_biserial": float("nan"),
            "evolution_ci_95": (float("nan"), float("nan")),
            "baseline_ci_95": (float("nan"), float("nan")),
        }

        evolution_scores = self.generation_scores
        if not evolution_scores or not baseline_scores:
            return nan_result

        n1, n2 = len(evolution_scores), len(baseline_scores)
        stat, p_value = stats.mannwhitneyu(
            evolution_scores, baseline_scores, alternative="greater"
        )

        # Rank-biserial correlation: positive when evolution tends to be higher.
        # r = (2*U)/(n1*n2) - 1: equals +1 when all evolution > all baseline.
        effect_size = float((2 * stat) / (n1 * n2) - 1)

        evo_ci = (
            (float(np.percentile(evolution_scores, 2.5)), float(np.percentile(evolution_scores, 97.5)))
            if n1 >= 3 else (float("nan"), float("nan"))
        )
        base_ci = (
            (float(np.percentile(baseline_scores, 2.5)), float(np.percentile(baseline_scores, 97.5)))
            if n2 >= 3 else (float("nan"), float("nan"))
        )

        return {
            "evolution_mean": float(np.mean(evolution_scores)),
            "baseline_mean": float(np.mean(baseline_scores)),
            "n_evolution": n1,
            "n_baseline": n2,
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size_rank_biserial": effect_size,
            "evolution_ci_95": evo_ci,
            "baseline_ci_95": base_ci,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def n_generations(self) -> int:
        """Number of generation results successfully loaded."""
        return len(self._records)

    @property
    def generation_scores(self) -> list[float]:
        """Aggregate best_score for each loaded generation, in generation order."""
        return [rec["best_score"] for rec in self._records]

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_records(self) -> list[dict]:
        """
        Load all gen_*/result.json files from the results directory.

        Silently skips missing or malformed JSON files with a warning. Returns
        records sorted by generation_id ascending.

        Returns:
            List of dicts (deserialized result.json contents), sorted by generation_id.
        """
        if not self._results_dir.exists():
            logger.warning("Results directory does not exist: %s", self._results_dir)
            return []

        records: list[dict] = []
        for gen_dir in sorted(self._results_dir.glob("gen_*")):
            result_file = gen_dir / "result.json"
            if not result_file.exists():
                logger.warning("Missing result.json in %s — skipping", gen_dir)
                continue
            try:
                with open(result_file) as fh:
                    data = json.load(fh)
                records.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s — skipping", result_file, exc)

        # Sort by generation_id to ensure consistent ordering.
        records.sort(key=lambda r: r.get("generation_id", 0))
        return records
