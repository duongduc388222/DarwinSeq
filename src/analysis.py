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
        and frequent LASSO retention.

        composite_score = selection_freq × retention_freq

        Also includes mean_coef from coefficient stats for additional ranking signal.

        Returns:
            DataFrame with columns:
              - gene (str): Gene symbol.
              - selection_freq (float): Fraction of generations in which LLM selected it.
              - retention_freq (float): Fraction of generations in which LASSO retained it.
              - composite_score (float): selection_freq × retention_freq.
              - mean_coef (float): Mean |LASSO coefficient| across all generations.
            Sorted by composite_score descending (ties broken by mean_coef).
            Empty DataFrame if no results loaded.
        """
        if not self._records:
            return pd.DataFrame(
                columns=["gene", "selection_freq", "retention_freq", "composite_score", "mean_coef"]
            )

        sel_df = self.compute_selection_frequency()
        ret_df = self.compute_retention_frequency()
        coef_df = self.compute_coefficient_stats()

        sel_map = dict(zip(sel_df["gene"], sel_df["selection_freq"]))
        ret_map = dict(zip(ret_df["gene"], ret_df["retention_freq"]))
        coef_map = dict(zip(coef_df["gene"], coef_df["mean_coef"]))

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
                    "mean_coef": coef_map.get(gene, 0.0),
                }
            )

        df = pd.DataFrame(
            rows,
            columns=["gene", "selection_freq", "retention_freq", "composite_score", "mean_coef"],
        )
        return (
            df.sort_values(["composite_score", "mean_coef"], ascending=[False, False])
            .reset_index(drop=True)
        )

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
