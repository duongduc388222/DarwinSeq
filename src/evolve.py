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

        Parses any per-generation checkpoint files written to the output_dir.
        Falls back to a single result from oe_result.best_program if no
        checkpoints are found.

        Args:
            oe_result: openevolve.api.EvolutionResult from run_evolution().

        Returns:
            List of GenerationResult ordered by generation_id.
        """
        results: list[GenerationResult] = []

        # Try per-generation checkpoint files first.
        checkpoint_files = sorted(self._output_dir.glob("checkpoint_gen_*.json"))
        if not checkpoint_files:
            # OpenEvolve may use a different naming pattern.
            checkpoint_files = sorted(self._output_dir.glob("checkpoint_*.json"))

        for i, ckpt_file in enumerate(checkpoint_files):
            try:
                gen_result = _parse_checkpoint(i, ckpt_file)
                results.append(gen_result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse checkpoint %s: %s", ckpt_file, exc)

        if not results and oe_result is not None:
            # Fall back: build one result from the best program.
            best_prog = getattr(oe_result, "best_program", None)
            artifacts = {}
            if best_prog is not None:
                artifacts = getattr(best_prog, "artifacts", {}) or {}
            raw_coefs = artifacts.get("coefficients", {})
            coefs = (
                {k: float(v) for k, v in raw_coefs.items()}
                if isinstance(raw_coefs, dict)
                else {}
            )
            results.append(
                GenerationResult(
                    generation_id=0,
                    best_score=float(getattr(oe_result, "best_score", 0.0)),
                    best_genes=list(artifacts.get("selected_genes", [])),
                    retained_genes=list(artifacts.get("retained_genes", [])),
                    coefficients=coefs,
                    all_scores=[float(getattr(oe_result, "best_score", 0.0))],
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
