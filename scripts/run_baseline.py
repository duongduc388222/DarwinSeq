"""
run_baseline.py — End-to-end baseline run for the DarwinSeq project.

Executes the full pipeline with random 200-gene subsets (100 in-vocab + 100
out-of-vocab) for N random seeds. No LLM guidance — this establishes the
random-selection performance baseline that the agentic system must beat.

Usage:
    python scripts/run_baseline.py --n_seeds 50 --output_dir results/baseline/

Output files:
    <output_dir>/all_runs.json     — per-seed results (list of dicts)
    <output_dir>/summary.json      — aggregate statistics across all seeds
    <output_dir>/best_run.json     — run with the highest aggregate_score
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader, DEFAULT_DATA_PATH, safe_load
from src.evaluator import LASSOEvaluator, DEFAULT_CONFIG_PATH
from src.gene_vocab import GeneVocabulary, DEFAULT_VOCAB_PATH
from src.sampler import CellSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core pipeline functions (importable for unit testing)
# ---------------------------------------------------------------------------


def run_single_seed(
    data_loader,
    vocab: GeneVocabulary,
    evaluator: LASSOEvaluator,
    seed: int,
    n_in: int = 100,
    n_out: int = 100,
    n_cells: int = 100,
) -> dict:
    """
    Run one full pipeline iteration for a given random seed.

    Samples n_in + n_out genes randomly, then samples n_cells cells, then
    evaluates LASSO performance. All randomness is controlled by seed.

    Args:
        data_loader: DataLoader instance (or compatible stub for testing).
        vocab: GeneVocabulary instance with in_vocab and out_vocab populated.
        evaluator: LASSOEvaluator instance.
        seed: Integer random seed controlling gene sampling and cell sampling.
        n_in: Number of in-vocab genes to sample.
        n_out: Number of out-of-vocab genes to sample.
        n_cells: Number of cells to sample.

    Returns:
        Dict with keys:
          - seed (int)
          - gene_list (list[str]): the n_in + n_out genes used
          - scores (dict[str, float]): per-target Pearson r
          - aggregate_score (float): mean Pearson r across targets
          - retained_genes (list[str]): genes with non-zero LASSO coefficient
          - n_retained (int): len(retained_genes)
    """
    gene_list = vocab.sample_subset(n_in=n_in, n_out=n_out, seed=seed)
    sampler = CellSampler(data_loader, gene_list, seed=seed)
    X, y = sampler.sample(n=n_cells)
    result = evaluator.evaluate(X, y)

    return {
        "seed": seed,
        "gene_list": gene_list,
        "scores": result.scores,
        "aggregate_score": result.aggregate_score,
        "retained_genes": result.retained_genes,
        "n_retained": result.n_retained,
    }


def compute_summary(all_runs: list[dict]) -> dict:
    """
    Compute aggregate statistics across all per-seed run results.

    Args:
        all_runs: List of dicts returned by run_single_seed().

    Returns:
        Dict with keys:
          - n_runs (int): number of runs
          - aggregate_score_mean (float)
          - aggregate_score_std (float)
          - aggregate_score_min (float)
          - aggregate_score_max (float)
          - gene_frequency (dict[str, int]): gene → number of runs in which
            it was retained (non-zero LASSO coefficient)
          - retained_count_distribution (list[int]): n_retained per run
    """
    if not all_runs:
        return {
            "n_runs": 0,
            "aggregate_score_mean": float("nan"),
            "aggregate_score_std": float("nan"),
            "aggregate_score_min": float("nan"),
            "aggregate_score_max": float("nan"),
            "gene_frequency": {},
            "retained_count_distribution": [],
        }

    agg_scores = [r["aggregate_score"] for r in all_runs]
    gene_frequency: dict[str, int] = {}
    retained_dist: list[int] = []

    for run in all_runs:
        retained_dist.append(run["n_retained"])
        for gene in run["retained_genes"]:
            gene_frequency[gene] = gene_frequency.get(gene, 0) + 1

    return {
        "n_runs": len(all_runs),
        "aggregate_score_mean": float(np.mean(agg_scores)),
        "aggregate_score_std": float(np.std(agg_scores)),
        "aggregate_score_min": float(np.min(agg_scores)),
        "aggregate_score_max": float(np.max(agg_scores)),
        "gene_frequency": gene_frequency,
        "retained_count_distribution": retained_dist,
    }


def select_best_run(all_runs: list[dict]) -> dict | None:
    """
    Return the run with the highest aggregate_score.

    Args:
        all_runs: List of dicts returned by run_single_seed().

    Returns:
        The dict with the maximum aggregate_score, or None if all_runs is empty.
    """
    if not all_runs:
        return None
    return max(all_runs, key=lambda r: r["aggregate_score"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    """
    Parse CLI arguments for run_baseline.py.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run the DarwinSeq baseline: random gene subsets over N seeds."
    )
    parser.add_argument(
        "--n_seeds", type=int, default=50,
        help="Number of random seeds to run (default: 50).",
    )
    parser.add_argument(
        "--n_in", type=int, default=100,
        help="In-vocab genes per seed (default: 100).",
    )
    parser.add_argument(
        "--n_out", type=int, default=100,
        help="Out-of-vocab genes per seed (default: 100).",
    )
    parser.add_argument(
        "--n_cells", type=int, default=100,
        help="Cells to sample per seed (default: 100).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/baseline/",
        help="Directory for output JSON files (default: results/baseline/).",
    )
    parser.add_argument(
        "--data_path", type=str, default=DEFAULT_DATA_PATH,
        help="Path to the .h5ad dataset file.",
    )
    parser.add_argument(
        "--vocab_path", type=str, default=DEFAULT_VOCAB_PATH,
        help="Path to the gene vocabulary text file.",
    )
    parser.add_argument(
        "--config_path", type=str, default=DEFAULT_CONFIG_PATH,
        help="Path to the LASSO hyperparameter JSON config.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """
    Load data, run N random-seed baseline iterations, and save results.

    Args:
        argv: CLI argument list. Defaults to sys.argv[1:].
    """
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", args.data_path)
    adata = safe_load(h5ad_path=args.data_path)
    data_loader = DataLoader(adata)

    logger.info("Building gene vocabulary from %s", args.vocab_path)
    vocab = GeneVocabulary(
        vocab_path=args.vocab_path,
        adata_var_names=list(adata.var_names),
    )
    vocab_info = vocab.validate()
    logger.info(
        "Vocabulary: %d in-vocab, %d out-of-vocab (coverage %.1f%%)",
        vocab_info["in_vocab_count"],
        vocab_info["out_vocab_count"],
        vocab_info["coverage_pct"],
    )

    evaluator = LASSOEvaluator(config_path=args.config_path)
    logger.info(
        "Starting baseline: %d seeds × (%d in + %d out genes) × %d cells",
        args.n_seeds, args.n_in, args.n_out, args.n_cells,
    )

    all_runs: list[dict] = []
    for i, seed in enumerate(range(args.n_seeds)):
        result = run_single_seed(
            data_loader=data_loader,
            vocab=vocab,
            evaluator=evaluator,
            seed=seed,
            n_in=args.n_in,
            n_out=args.n_out,
            n_cells=args.n_cells,
        )
        all_runs.append(result)
        logger.info(
            "Seed %3d/%d  aggregate=%.4f  retained=%d",
            i + 1, args.n_seeds,
            result["aggregate_score"],
            result["n_retained"],
        )

    # Save all per-seed results.
    all_runs_path = output_dir / "all_runs.json"
    with open(all_runs_path, "w") as fh:
        json.dump(all_runs, fh, indent=2)
    logger.info("Saved all runs → %s", all_runs_path)

    # Compute and save summary statistics.
    summary = compute_summary(all_runs)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(
        "Summary: mean=%.4f ± %.4f  [%.4f, %.4f]",
        summary["aggregate_score_mean"],
        summary["aggregate_score_std"],
        summary["aggregate_score_min"],
        summary["aggregate_score_max"],
    )
    logger.info("Saved summary → %s", summary_path)

    # Save the best-performing run.
    best = select_best_run(all_runs)
    if best is not None:
        best_path = output_dir / "best_run.json"
        with open(best_path, "w") as fh:
            json.dump(best, fh, indent=2)
        logger.info(
            "Best run: seed=%d  aggregate=%.4f  retained=%d → %s",
            best["seed"], best["aggregate_score"], best["n_retained"], best_path,
        )


if __name__ == "__main__":
    main()
