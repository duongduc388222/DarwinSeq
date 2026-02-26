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
        best_score: Aggregate balanced accuracy of the best individual in this generation.
        best_genes: The 200-gene list selected by the best individual.
        retained_genes: Genes with non-zero LASSO coefficient in the best individual.
        coefficients: Gene → summed absolute LASSO coefficient for the best individual.
                      Empty dict if coefficients were not recorded.
        all_scores: Aggregate scores for all evaluated individuals this generation.
        timestamp: ISO 8601 UTC timestamp when the generation completed.
    """

    generation_id: int
    best_score: float
    best_genes: list = field(default_factory=list)
    retained_genes: list = field(default_factory=list)
    coefficients: dict = field(default_factory=dict)
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

        Uses openevolve.run_evolution() with the DarwinSeq evaluator adapter.
        Calls log_generation for each GenerationResult after the run completes.

        Args:
            n_generations: Number of generations to run.  If None, uses the value
                           from config['evolution']['max_iterations'].

        Returns:
            List of GenerationResult, one per generation (or one for the whole
            run if per-generation checkpoints are not available).
        """
        from openevolve import run_evolution  # imported here to allow tests without openevolve

        oe_config = self._build_openevolve_config(n_generations)
        n_iters = oe_config.max_iterations

        project_root = Path(__file__).parent.parent
        paths = self._config.get("paths", {})
        program_file = str(
            project_root / paths.get("program_file", "src/gene_selector_template.py")
        )
        evaluator_file = str(Path(__file__).parent / "openevolve_adapter.py")

        logger.info(
            "Starting evolution: %d iterations, program=%s", n_iters, program_file
        )

        oe_result = run_evolution(
            initial_program=program_file,
            evaluator=evaluator_file,
            config=oe_config,
            iterations=n_iters,
            output_dir=str(self._output_dir),
            cleanup=False,
        )

        results = self._collect_results(oe_result)
        for gen_result in results:
            self.log_generation(gen_result.generation_id, gen_result)

        best_score = max((r.best_score for r in results), default=float("nan"))
        logger.info("Evolution complete. Best score: %.4f", best_score)
        return results

    def log_generation(self, gen_id: int, result: GenerationResult) -> None:
        """
        Persist a GenerationResult to disk as JSON.

        Creates results/evolution/gen_{gen_id}/ and writes three files:
          - result.json: Full GenerationResult serialized as dict.
          - selected_genes.json: The 200-gene list from result.best_genes.
          - eval_result.json: Compact evaluation summary with aggregate_score,
            retained_genes, coefficients, and n_retained.

        Args:
            gen_id: Generation index (used for the directory name).
            result: GenerationResult to serialize.
        """
        gen_dir = self._output_dir / f"gen_{gen_id}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Full result (all fields)
        out_path = gen_dir / "result.json"
        with open(out_path, "w") as fh:
            json.dump(asdict(result), fh, indent=2)

        # Selected genes list (for easy loading by analysis module)
        selected_path = gen_dir / "selected_genes.json"
        with open(selected_path, "w") as fh:
            json.dump(result.best_genes, fh, indent=2)

        # Compact evaluation summary
        eval_path = gen_dir / "eval_result.json"
        with open(eval_path, "w") as fh:
            json.dump(
                {
                    "aggregate_score": result.best_score,
                    "retained_genes": result.retained_genes,
                    "coefficients": result.coefficients,
                    "n_retained": len(result.retained_genes),
                },
                fh,
                indent=2,
            )

        logger.info("Generation %d logged to %s", gen_id, gen_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_openevolve_config(self, n_generations: int | None):
        """
        Build an openevolve Config object from the loaded YAML.

        Translates our evolve_config.yaml structure into the openevolve v0.2.26+
        Config dataclass, expanding any ${VAR} environment variable placeholders.

        Args:
            n_generations: Optional override for max_iterations.

        Returns:
            openevolve.config.Config instance ready to pass to run_evolution().
        """
        from openevolve.config import Config, LLMModelConfig

        cfg = self._config
        evo = cfg.get("evolution", {})
        llm_cfg = cfg.get("llm", {})

        config = Config()

        # Evolution settings
        config.max_iterations = n_generations if n_generations is not None else evo.get(
            "max_iterations", 5
        )
        config.checkpoint_interval = evo.get("checkpoint_interval", 1)
        if evo.get("random_seed") is not None:
            config.random_seed = evo["random_seed"]
            config.database.random_seed = evo["random_seed"]

        # LLM model — expand ${VAR} placeholders using os.environ
        api_key = _resolve_env_var(str(llm_cfg.get("api_key", "")))
        model = LLMModelConfig(
            name=llm_cfg.get("primary_model", "gpt-4"),
            api_base=llm_cfg.get("api_base"),
            api_key=api_key or None,
            weight=float(llm_cfg.get("primary_weight", 1.0)),
            temperature=float(llm_cfg.get("temperature", 0.7)),
            max_tokens=int(llm_cfg.get("max_tokens", 4096)),
            timeout=int(llm_cfg["timeout"]) if llm_cfg.get("timeout") is not None else 300,
            retries=int(llm_cfg["retries"]) if llm_cfg.get("retries") is not None else 5,
        )
        config.llm.models = [model]

        # Population / database settings
        config.database.population_size = evo.get("population_size", 5)
        config.database.archive_size = evo.get("archive_size", 3)
        config.database.num_islands = evo.get("num_islands", 1)

        # System message (inline in YAML under 'system_message' key)
        if cfg.get("system_message"):
            config.prompt.system_message = cfg["system_message"]

        return config

    def _collect_results(self, oe_result) -> list[GenerationResult]:
        """
        Build a GenerationResult list from an openevolve EvolutionResult.

        Search order (stops at first hit):
          1. Per-generation checkpoint directories: checkpoints/checkpoint_{N}/
             (OpenEvolve v0.2+ format — one directory per iteration)
          2. Per-generation checkpoint JSON files (checkpoint_gen_*.json or checkpoint_*.json)
             (older OpenEvolve pattern)
          3. best/best_program_info.json — written by OpenEvolve v0.2+ after the run
          4. oe_result.best_program attributes fallback (last resort)

        Args:
            oe_result: openevolve.api.EvolutionResult from run_evolution().

        Returns:
            List of GenerationResult ordered by generation_id.
        """
        results: list[GenerationResult] = []

        # 1. Scan checkpoints/checkpoint_{iteration}/ directories (OpenEvolve v0.2+ format).
        #    Each directory contains metadata.json (best_program_id) and
        #    programs/{id}.json (metrics + artifacts_json with retained_genes, coefficients).
        ckpt_root = self._output_dir / "checkpoints"
        if ckpt_root.is_dir():
            ckpt_dirs = sorted(
                [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
                key=lambda p: _parse_checkpoint_iteration(p.name),
            )
            for i, ckpt_dir in enumerate(ckpt_dirs):
                try:
                    gen_result = _parse_checkpoint_dir(i, ckpt_dir)
                    if gen_result is not None:
                        results.append(gen_result)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to parse checkpoint dir %s: %s", ckpt_dir, exc)

        # 2. Try per-generation checkpoint JSON files (older OpenEvolve pattern).
        if not results:
            checkpoint_files = sorted(self._output_dir.glob("checkpoint_gen_*.json"))
            if not checkpoint_files:
                checkpoint_files = sorted(self._output_dir.glob("checkpoint_*.json"))

            for i, ckpt_file in enumerate(checkpoint_files):
                try:
                    gen_result = _parse_checkpoint(i, ckpt_file)
                    results.append(gen_result)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to parse checkpoint %s: %s", ckpt_file, exc)

        # 3. Try best/best_program_info.json (OpenEvolve v0.2+ saves this after the run).
        if not results:
            best_info_path = self._output_dir / "best" / "best_program_info.json"
            if best_info_path.exists():
                try:
                    results.append(_parse_best_program_info(0, best_info_path))
                    logger.info("Loaded result from %s", best_info_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to parse %s: %s", best_info_path, exc)

        # 4. Last-resort fallback: read from oe_result object attributes.
        if not results and oe_result is not None:
            best_prog = getattr(oe_result, "best_program", None)
            artifacts = {}
            if best_prog is not None:
                # Try metrics dict on the program object first.
                prog_metrics = getattr(best_prog, "metrics", {}) or {}
                artifacts = getattr(best_prog, "artifacts", {}) or {}
                score = float(
                    prog_metrics.get("primary")
                    or prog_metrics.get("balanced_accuracy")
                    or 0.0
                )
            else:
                prog_metrics = {}
                score = 0.0

            raw_coefs = artifacts.get("coefficients", {})
            coefs = (
                {k: float(v) for k, v in raw_coefs.items()}
                if isinstance(raw_coefs, dict)
                else {}
            )
            results.append(
                GenerationResult(
                    generation_id=0,
                    best_score=score,
                    best_genes=list(artifacts.get("selected_genes", [])),
                    retained_genes=list(artifacts.get("retained_genes", [])),
                    coefficients=coefs,
                    all_scores=[score],
                    timestamp=_utc_now(),
                )
            )

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_env_var(value: str) -> str:
    """
    Expand ``${VAR}`` placeholders in a string using os.environ.

    Args:
        value: String possibly containing ``${VAR_NAME}`` patterns.

    Returns:
        String with placeholders replaced by their environment variable values.
        Unresolved placeholders are left unchanged.
    """
    import re

    return re.sub(
        r"\$\{([^}]+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def _utc_now() -> str:
    """
    Return the current UTC time as an ISO 8601 string.

    Returns:
        UTC timestamp string, e.g. '2026-02-26T12:00:00+00:00'.
    """
    return datetime.now(timezone.utc).isoformat()


def generate_summary(
    results: list,
    baseline_mean_score: float | None = None,
) -> dict:
    """
    Build a JSON-serializable summary dict from a list of GenerationResult objects.

    Args:
        results: Per-generation results from EvolutionRunner.run().
        baseline_mean_score: Optional random-baseline aggregate score for comparison.

    Returns:
        Dict with keys:
            - generations: list of per-gen dicts {gen, aggregate_score, n_retained, top_genes}
            - best_generation: int index of the highest-scoring generation (None if empty)
            - best_score: float, the highest aggregate_score seen (None if empty)
            - baseline_mean_score: float or None
    """
    if not results:
        return {
            "generations": [],
            "best_generation": None,
            "best_score": None,
            "baseline_mean_score": baseline_mean_score,
        }

    gen_summaries = []
    for r in results:
        if r.coefficients:
            top_genes = sorted(
                r.retained_genes,
                key=lambda g: r.coefficients.get(g, 0.0),
                reverse=True,
            )[:10]
        else:
            top_genes = list(r.retained_genes[:10])

        gen_summaries.append({
            "gen": r.generation_id,
            "aggregate_score": r.best_score,
            "n_retained": len(r.retained_genes),
            "top_genes": top_genes,
        })

    best = max(results, key=lambda r: r.best_score)
    return {
        "generations": gen_summaries,
        "best_generation": best.generation_id,
        "best_score": best.best_score,
        "baseline_mean_score": baseline_mean_score,
    }


def _parse_best_program_info(gen_id: int, info_path: Path) -> GenerationResult:
    """
    Parse an OpenEvolve best/best_program_info.json file into a GenerationResult.

    OpenEvolve v0.2+ writes this file after the run with the best program's
    metrics and artifacts. The score is taken from metrics['primary'] (or
    metrics['balanced_accuracy'] as fallback) — NOT from oe_result.best_score,
    which is an internal OpenEvolve counter unrelated to our fitness metric.

    Args:
        gen_id: Generation index to assign to the result.
        info_path: Path to best_program_info.json.

    Returns:
        GenerationResult populated from the info file.
    """
    with open(info_path) as fh:
        data = json.load(fh)

    metrics = data.get("metrics", {}) or {}
    artifacts = data.get("artifacts", {}) or {}

    score = float(
        metrics.get("primary")
        or metrics.get("balanced_accuracy")
        or metrics.get("aggregate_score", 0.0)
        or 0.0
    )

    raw_coefs = artifacts.get("coefficients", {})
    coefficients = (
        {k: float(v) for k, v in raw_coefs.items()}
        if isinstance(raw_coefs, dict)
        else {}
    )

    return GenerationResult(
        generation_id=gen_id,
        best_score=score,
        best_genes=list(artifacts.get("selected_genes", [])),
        retained_genes=list(artifacts.get("retained_genes", [])),
        coefficients=coefficients,
        all_scores=[score],
        timestamp=data.get("timestamp") or _utc_now(),
    )


def _parse_checkpoint_iteration(dir_name: str) -> int:
    """
    Extract the iteration number from a checkpoint directory name.

    OpenEvolve names checkpoint directories as 'checkpoint_{N}' where N is
    the iteration number.  Returns 0 for any name that doesn't match.

    Args:
        dir_name: Directory name, e.g. 'checkpoint_3'.

    Returns:
        Parsed integer iteration, or 0 on failure.
    """
    try:
        return int(dir_name.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 0


def _parse_checkpoint_dir(gen_id: int, ckpt_dir: Path) -> GenerationResult | None:
    """
    Parse an OpenEvolve v0.2+ checkpoint directory into a GenerationResult.

    Each checkpoint directory (checkpoints/checkpoint_{N}/) contains:
      - metadata.json: has best_program_id, last_iteration
      - programs/{id}.json: full Program with metrics + artifacts_json
      - best_program_info.json: metrics only (no artifacts — used as fallback)

    Artifacts (retained_genes, coefficients, selected_genes) are stored in
    programs/{id}.json under the artifacts_json field (JSON-serialised dict).

    Args:
        gen_id: Zero-based index to assign as the generation_id.
        ckpt_dir: Path to the checkpoint directory.

    Returns:
        GenerationResult populated from the checkpoint, or None on critical failure.
    """
    # Step 1: Read metadata to find the best_program_id.
    metadata_path = ckpt_dir / "metadata.json"
    best_program_id: str | None = None

    if metadata_path.exists():
        try:
            with open(metadata_path) as fh:
                meta = json.load(fh)
            best_program_id = meta.get("best_program_id")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read %s: %s", metadata_path, exc)

    # Step 2: Read best_program_info.json for score + timestamp.
    best_info_path = ckpt_dir / "best_program_info.json"
    score = 0.0
    timestamp = _utc_now()

    if best_info_path.exists():
        try:
            with open(best_info_path) as fh:
                info = json.load(fh)
            metrics = info.get("metrics", {}) or {}
            score = float(
                metrics.get("primary")
                or metrics.get("balanced_accuracy")
                or metrics.get("aggregate_score", 0.0)
                or 0.0
            )
            timestamp = info.get("timestamp") or timestamp
            if best_program_id is None:
                best_program_id = info.get("id")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read %s: %s", best_info_path, exc)

    # Step 3: Load the full program JSON for artifacts (retained_genes, coefficients, …).
    artifacts: dict = {}
    if best_program_id:
        program_path = ckpt_dir / "programs" / f"{best_program_id}.json"
        if program_path.exists():
            try:
                with open(program_path) as fh:
                    prog_data = json.load(fh)

                # Override score from the program metrics if available.
                prog_metrics = prog_data.get("metrics", {}) or {}
                prog_score = float(
                    prog_metrics.get("primary")
                    or prog_metrics.get("balanced_accuracy")
                    or prog_metrics.get("aggregate_score", 0.0)
                    or 0.0
                )
                if prog_score > 0:
                    score = prog_score

                # Parse artifacts_json (small artifacts stored inline as JSON string).
                artifacts_json = prog_data.get("artifacts_json")
                if artifacts_json:
                    try:
                        artifacts = json.loads(artifacts_json)
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.debug("Could not decode artifacts_json in %s: %s", program_path, exc)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not read program file %s: %s", program_path, exc)

    raw_coefs = artifacts.get("coefficients", {})
    coefficients = (
        {k: float(v) for k, v in raw_coefs.items()}
        if isinstance(raw_coefs, dict)
        else {}
    )

    return GenerationResult(
        generation_id=gen_id,
        best_score=score,
        best_genes=list(artifacts.get("selected_genes", [])),
        retained_genes=list(artifacts.get("retained_genes", [])),
        coefficients=coefficients,
        all_scores=[score],
        timestamp=timestamp,
    )


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

    raw_coefs = artifacts.get("coefficients", {})
    coefficients = {k: float(v) for k, v in raw_coefs.items()} if isinstance(raw_coefs, dict) else {}

    return GenerationResult(
        generation_id=gen_id,
        best_score=fitness,
        best_genes=list(artifacts.get("selected_genes", [])),
        retained_genes=list(artifacts.get("retained_genes", [])),
        coefficients=coefficients,
        all_scores=all_scores,
        timestamp=data.get("timestamp") or _utc_now(),
    )
