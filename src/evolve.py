"""
evolve.py — EvolutionRunner orchestrator for OpenEvolve-based gene selection.

Provides EvolutionRunner, a high-level wrapper around OpenEvolve's Evolver that:
  - Loads the evolve_config.yaml
  - Configures and runs OpenEvolve with the DarwinSeq pipeline
  - Collects per-generation results into GenerationResult dataclasses
  - Logs each generation's output to results/evolution/gen_{id}/

Usage example:
    runner = EvolutionRunner("config/evolve_config.yaml")
    results = runner.run(n_generations=5)
    for r in results:
        print(f"Gen {r.generation_id}: score={r.best_score:.4f}")
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Load .env file so OPENAI_API_KEY (and other secrets) are available as
# environment variables before OpenEvolve reads them via ${VAR} expansion.
try:
    from dotenv import load_dotenv as _load_dotenv

    _dotenv_path = Path(__file__).parent.parent / ".env"
    _load_dotenv(_dotenv_path, override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on env vars being set externally

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """
    Summary of results from a single evolution generation.

    Attributes:
        generation_id: Zero-based generation index.
        best_score: Aggregate Pearson r of the best individual in this generation.
        best_genes: The 200-gene list selected by the best individual.
        retained_genes: Genes with non-zero LASSO coefficient in the best individual.
        all_scores: Aggregate scores for all evaluated individuals this generation.
        timestamp: ISO 8601 UTC timestamp when the generation completed.
    """

    generation_id: int
    best_score: float
    best_genes: list = field(default_factory=list)
    retained_genes: list = field(default_factory=list)
    all_scores: list = field(default_factory=list)
    timestamp: str = ""


class EvolutionRunner:
    """
    High-level orchestrator for OpenEvolve-based gene selection evolution.

    Loads evolve_config.yaml, configures OpenEvolve, runs the evolution loop,
    and persists per-generation results to disk.

    Args:
        config_path: Path to the evolve_config.yaml file.
        output_dir: Override the output directory from the config.  If None,
                    uses the value from config['paths']['output_dir'].
    """

    def __init__(self, config_path: str, output_dir: str | None = None) -> None:
        config_path = str(config_path)
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Evolution config not found: {config_path}")

        with open(config_path) as fh:
            self._config: dict = yaml.safe_load(fh)

        self._output_dir = Path(
            output_dir or self._config.get("paths", {}).get("output_dir", "results/evolution")
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, n_generations: int | None = None) -> list[GenerationResult]:
        """
        Run the full evolution loop via OpenEvolve and return per-generation results.

        Overrides max_iterations in the config if n_generations is given.  After
        OpenEvolve finishes, loads checkpoint files to reconstruct per-generation
        results and calls log_generation for each one.

        Args:
            n_generations: Number of generations to run.  If None, uses the value
                           from config['evolution']['max_iterations'].

        Returns:
            List of GenerationResult, one per generation, ordered by generation_id.
        """
        from openevolve import Evolver  # imported here to allow tests without openevolve

        config = self._build_openevolve_config(n_generations)
        logger.info(
            "Starting evolution: %d generations × pop=%d",
            config["evolution"]["max_iterations"],
            config["evolution"]["population_size"],
        )

        evolver = Evolver(config)
        evolver.run()

        results = self._collect_results(evolver, config)
        for gen_result in results:
            self.log_generation(gen_result.generation_id, gen_result)

        logger.info(
            "Evolution complete. Best score: %.4f (gen %d)",
            max((r.best_score for r in results), default=float("nan")),
            max(
                (r.generation_id for r in results if r.best_score == max(
                    (x.best_score for x in results), default=float("nan")
                )),
                default=-1,
            ),
        )
        return results

    def log_generation(self, gen_id: int, result: GenerationResult) -> None:
        """
        Persist a GenerationResult to disk as JSON.

        Creates results/evolution/gen_{gen_id}/ and writes result.json inside it.

        Args:
            gen_id: Generation index (used for the directory name).
            result: GenerationResult to serialize.
        """
        gen_dir = self._output_dir / f"gen_{gen_id}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        out_path = gen_dir / "result.json"
        with open(out_path, "w") as fh:
            json.dump(asdict(result), fh, indent=2)

        logger.info("Generation %d logged to %s", gen_id, out_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_openevolve_config(self, n_generations: int | None) -> dict:
        """
        Build the OpenEvolve config dict from the loaded YAML.

        Overrides max_iterations if n_generations is provided.  Resolves the
        program_file and evaluator paths to absolute paths so OpenEvolve can
        locate them regardless of working directory.

        Args:
            n_generations: Optional override for max_iterations.

        Returns:
            Config dict ready to pass to openevolve.Evolver.
        """
        config = _deep_copy_dict(self._config)

        if n_generations is not None:
            config.setdefault("evolution", {})["max_iterations"] = n_generations

        # Resolve file paths relative to project root (parent of config/).
        project_root = Path(__file__).parent.parent
        paths = config.get("paths", {})
        for key in ("program_file", "output_dir", "checkpoint_dir"):
            if key in paths and not Path(paths[key]).is_absolute():
                paths[key] = str(project_root / paths[key])

        return config

    def _collect_results(self, evolver, config: dict) -> list[GenerationResult]:
        """
        Collect per-generation results from OpenEvolve's checkpoint directory.

        Reads checkpoint JSON files written by OpenEvolve during evolution and
        converts them to GenerationResult objects.  Falls back to a single
        result from the best individual if no checkpoints are found.

        Args:
            evolver: Completed openevolve.Evolver instance.
            config: The config dict that was used to run evolution.

        Returns:
            List of GenerationResult ordered by generation_id.
        """
        checkpoint_dir = Path(
            config.get("paths", {}).get("checkpoint_dir", "results/checkpoints")
        )

        results: list[GenerationResult] = []

        if checkpoint_dir.exists():
            checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.json"))
            for i, ckpt_file in enumerate(checkpoint_files):
                try:
                    gen_result = _parse_checkpoint(i, ckpt_file)
                    results.append(gen_result)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to parse checkpoint %s: %s", ckpt_file, exc)

        # If no checkpoints were found, synthesize a single result from the best
        # individual returned by the evolver.
        if not results:
            try:
                best = evolver.get_best_individual()
                results.append(
                    GenerationResult(
                        generation_id=0,
                        best_score=float(getattr(best, "fitness", 0.0)),
                        best_genes=list(
                            getattr(best, "artifacts", {}).get("selected_genes", [])
                        ),
                        retained_genes=list(
                            getattr(best, "artifacts", {}).get("retained_genes", [])
                        ),
                        all_scores=[float(getattr(best, "fitness", 0.0))],
                        timestamp=_utc_now(),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not retrieve best individual: %s", exc)

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────


def _deep_copy_dict(d: dict) -> dict:
    """
    Return a deep copy of a plain dict/list/scalar structure via JSON round-trip.

    Args:
        d: Dictionary to copy (must be JSON-serializable).

    Returns:
        Deep copy of d.
    """
    return json.loads(json.dumps(d))


def _utc_now() -> str:
    """
    Return the current UTC time as an ISO 8601 string.

    Returns:
        UTC timestamp string, e.g. '2026-02-26T12:00:00+00:00'.
    """
    return datetime.now(timezone.utc).isoformat()


def _parse_checkpoint(gen_id: int, checkpoint_path: Path) -> GenerationResult:
    """
    Parse an OpenEvolve checkpoint JSON file into a GenerationResult.

    OpenEvolve's checkpoint format may vary across versions; this function
    handles the common fields and falls back gracefully for missing ones.

    Args:
        gen_id: Generation index to assign to the result.
        checkpoint_path: Path to the checkpoint JSON file.

    Returns:
        GenerationResult populated from the checkpoint data.
    """
    with open(checkpoint_path) as fh:
        data = json.load(fh)

    # OpenEvolve checkpoints may store results under different keys depending
    # on version.  Try common patterns.
    best_individual = (
        data.get("best_individual")
        or data.get("best")
        or (data.get("population", [{}]) or [{}])[0]
    )

    fitness = float(
        best_individual.get("fitness")
        or best_individual.get("score")
        or best_individual.get("metrics", {}).get("primary", 0.0)
        or 0.0
    )

    artifacts = best_individual.get("artifacts") or best_individual.get("metadata") or {}
    all_individuals = data.get("population") or data.get("individuals") or [best_individual]
    all_scores = [
        float(ind.get("fitness") or ind.get("score") or ind.get("metrics", {}).get("primary", 0.0) or 0.0)
        for ind in all_individuals
    ]

    return GenerationResult(
        generation_id=gen_id,
        best_score=fitness,
        best_genes=list(artifacts.get("selected_genes", [])),
        retained_genes=list(artifacts.get("retained_genes", [])),
        all_scores=all_scores,
        timestamp=data.get("timestamp") or _utc_now(),
    )
