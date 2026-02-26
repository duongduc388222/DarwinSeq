"""
run_evolution.py — CLI script to run OpenEvolve-based gene selection evolution.

Usage:
    python scripts/run_evolution.py [OPTIONS]

Options:
    --config CONFIG           Path to evolve_config.yaml  [default: config/evolve_config.yaml]
    --n_generations N         Number of generations to run  [default: 5]
    --output_dir DIR          Directory for evolution output  [default: results/evolution]
    --baseline_score SCORE    Baseline aggregate score for comparison  [default: None]
    --data_path PATH          Path to the SEAAD A9 .h5ad file  [default: value in src/data_loader.py]
    --vocab_path PATH         Path to gene_vocabulary.txt  [default: config/gene_vocabulary.txt]

Requires openevolve to be installed:
    pip install openevolve>=0.2.0

Environment:
    OPENAI_API_KEY       — Gemini API key (loaded from .env automatically by src/evolve.py)
    DARWINSEQ_DATA_PATH  — Override data path (set automatically from --data_path)
    DARWINSEQ_VOCAB_PATH — Override vocab path (set automatically from --vocab_path)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path so src.* imports work regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolve import EvolutionRunner, generate_summary  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    """
    Parse CLI arguments for the evolution run script.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Namespace with attributes: config, n_generations, output_dir, baseline_score,
        data_path, vocab_path.
    """
    parser = argparse.ArgumentParser(
        description="Run OpenEvolve gene selection evolution for DarwinSeq.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/evolve_config.yaml",
        help="Path to evolve_config.yaml",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=5,
        help="Number of generations to run",
    )
    parser.add_argument(
        "--output_dir",
        default="results/evolution",
        help="Directory for evolution output (gen_*/ subdirs and summary.json)",
    )
    parser.add_argument(
        "--baseline_score",
        type=float,
        default=None,
        help="Random-baseline aggregate score for comparison in summary",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help=(
            "Path to the SEAAD A9 .h5ad file. "
            "Overrides the default in src/data_loader.py and the "
            "DARWINSEQ_DATA_PATH environment variable."
        ),
    )
    parser.add_argument(
        "--vocab_path",
        default=None,
        help=(
            "Path to gene_vocabulary.txt. "
            "Overrides the default config/gene_vocabulary.txt and the "
            "DARWINSEQ_VOCAB_PATH environment variable."
        ),
    )
    return parser.parse_args(argv)


def _ensure_openevolve() -> None:
    """
    Check that openevolve is installed, exiting with a helpful message if not.

    Raises:
        SystemExit: If openevolve cannot be imported.
    """
    try:
        import openevolve  # noqa: F401
    except ImportError:
        print(
            "ERROR: openevolve is not installed.\n"
            "  Install it with:  pip install openevolve>=0.2.0\n"
            "  Then set your Gemini API key in .env:\n"
            "    OPENAI_API_KEY=your_gemini_api_key_here",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_summary(summary: dict) -> None:
    """
    Print a formatted summary table of evolution results to stdout.

    Args:
        summary: Dict returned by generate_summary().
    """
    print("\n=== Evolution Summary ===")
    if summary["best_score"] is not None:
        print(
            f"Best score:     {summary['best_score']:.4f}"
            f"  (generation {summary['best_generation']})"
        )
    if summary.get("baseline_mean_score") is not None:
        print(f"Baseline score: {summary['baseline_mean_score']:.4f}")
    if summary.get("elapsed_seconds") is not None:
        print(f"Elapsed:        {summary['elapsed_seconds']:.1f}s")

    generations = summary.get("generations", [])
    if not generations:
        print("No generation data available.")
        return

    print(f"\n{'Gen':>4}  {'Score':>8}  {'Retained':>8}  Top genes")
    print("-" * 70)
    for gen in generations:
        top = ", ".join(gen["top_genes"][:5]) if gen["top_genes"] else "—"
        score = gen["aggregate_score"]
        score_str = f"{score:.4f}" if score is not None else "   N/A"
        print(f"{gen['gen']:>4}  {score_str:>8}  {gen['n_retained']:>8}  {top}")


def main(argv=None) -> None:
    """
    Entry point: parse args, guard for openevolve, run evolution, write summary.

    Args:
        argv: Argument list passed to parse_args (defaults to sys.argv[1:]).
    """
    args = parse_args(argv)

    _ensure_openevolve()

    # Export data/vocab paths as env vars so the evaluator adapter (imported
    # dynamically by OpenEvolve) can pick them up via _get_singleton_evaluator().
    if args.data_path:
        os.environ["DARWINSEQ_DATA_PATH"] = args.data_path
    if args.vocab_path:
        os.environ["DARWINSEQ_VOCAB_PATH"] = args.vocab_path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:      {args.config}")
    print(f"Generations: {args.n_generations}")
    print(f"Output dir:  {output_dir}")
    if args.data_path:
        print(f"Data path:   {args.data_path}")
    if args.vocab_path:
        print(f"Vocab path:  {args.vocab_path}")
    if args.baseline_score is not None:
        print(f"Baseline:    {args.baseline_score:.4f}")
    print()

    runner = EvolutionRunner(args.config, output_dir=str(output_dir))

    t0 = time.perf_counter()
    results = runner.run(n_generations=args.n_generations)
    elapsed = time.perf_counter() - t0

    summary = generate_summary(results, baseline_mean_score=args.baseline_score)
    summary["elapsed_seconds"] = round(elapsed, 2)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    _print_summary(summary)
    print(f"\nEvolution complete in {elapsed:.1f}s.")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
