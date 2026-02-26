"""
openevolve_adapter.py — Adapter connecting DarwinSeq's pipeline to OpenEvolve's
evaluator interface.

OpenEvolve v0.2.26+ imports this file and calls the module-level
``evaluate_stage1(program_path)`` function each generation.  The function:
  1. Loads ``select_genes`` from the evolved program file
  2. Delegates to GeneSelectorEvaluator.evaluate_stage1(select_genes_func)
  3. Returns EvaluationResult with fitness (balanced accuracy) and LLM artifacts

GeneSelectorEvaluator is also useful for unit tests (pass the function directly).
Heavy objects (DataLoader, GeneVocabulary, ADNCEvaluator) are loaded lazily on
the first call so tests can instantiate the class without a real h5ad file.
"""

import importlib.util
import logging
import sys
import traceback
from pathlib import Path

# Add project root to sys.path so openevolve_adapter can be imported from any
# working directory (OpenEvolve may run from a different cwd).
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from openevolve.evaluation_result import EvaluationResult

from src.data_loader import DEFAULT_DATA_PATH, DataLoader
from src.evaluator import DEFAULT_CONFIG_PATH, ADNCEvaluator
from src.gene_vocab import DEFAULT_VOCAB_PATH, GeneVocabulary
from src.sampler import CellSampler

logger = logging.getLogger(__name__)

# ── Module-level singleton used by the evaluate_stage1 entry point ────────────
# OpenEvolve imports this file once per process and calls evaluate_stage1
# for each evolved program.  A singleton avoids reloading the h5ad file.
_singleton_evaluator: "GeneSelectorEvaluator | None" = None


def _get_singleton_evaluator() -> "GeneSelectorEvaluator":
    """
    Return the module-level GeneSelectorEvaluator singleton, creating it on first call.

    Reads DARWINSEQ_DATA_PATH and DARWINSEQ_VOCAB_PATH environment variables so
    that scripts/run_evolution.py can pass custom paths without modifying this file.

    Returns:
        Shared GeneSelectorEvaluator instance.
    """
    import os

    global _singleton_evaluator
    if _singleton_evaluator is None:
        data_path = os.environ.get("DARWINSEQ_DATA_PATH") or None
        vocab_path = os.environ.get("DARWINSEQ_VOCAB_PATH") or None
        _singleton_evaluator = GeneSelectorEvaluator(
            data_path=data_path,
            vocab_path=vocab_path,
        )
    return _singleton_evaluator


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """
    Module-level entry point called by OpenEvolve for each evolved program.

    OpenEvolve writes the evolved program to a temp file and calls this function
    with its path.  We load ``select_genes`` from that file and pass it to the
    GeneSelectorEvaluator pipeline.

    Args:
        program_path: Absolute path to the evolved Python program file.

    Returns:
        EvaluationResult with:
            metrics = {'primary': balanced_accuracy, 'balanced_accuracy': float, ...}
            artifacts = {'retained_genes': [...], 'coefficients': {...}, ...}
    """
    try:
        spec = importlib.util.spec_from_file_location("_evolved_program", program_path)
        if spec is None or spec.loader is None:
            return EvaluationResult(
                metrics={"primary": 0.0},
                artifacts={"error": True, "error_message": f"Cannot load: {program_path}"},
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "select_genes"):
            return EvaluationResult(
                metrics={"primary": 0.0},
                artifacts={
                    "error": True,
                    "error_message": "select_genes function not found in evolved program",
                },
            )

        return _get_singleton_evaluator().evaluate_stage1(module.select_genes)

    except Exception as exc:
        err_msg = f"evaluate_stage1 failed loading {program_path}: {exc}\n{traceback.format_exc()}"
        logger.error(err_msg)
        return EvaluationResult(
            metrics={"primary": 0.0},
            artifacts={"error": True, "error_message": err_msg},
        )


# OpenEvolve requires a module-level ``evaluate`` function to pass the
# _load_evaluation_function check at controller init time.
# ``evaluate_stage1`` is also present so OpenEvolve uses the cascade path when
# cascade_evaluation is enabled.
evaluate = evaluate_stage1

# Default number of cells to sample per evaluation run.
DEFAULT_N_CELLS = 100


class GeneSelectorEvaluator:
    """
    OpenEvolve evaluator that wraps the DarwinSeq pipeline.

    Provides evaluate_stage1(select_genes_func) which OpenEvolve calls each
    generation with the evolved function.

    Args:
        data_path: Path to the SEAAD A9 h5ad file. Defaults to DEFAULT_DATA_PATH.
        vocab_path: Path to the gene vocabulary text file. Defaults to DEFAULT_VOCAB_PATH.
        config_path: Path to the model JSON config. Defaults to DEFAULT_CONFIG_PATH.
        n_cells: Number of cells to sample per evaluation. Default 100.
        sample_seed: Random seed for cell sampling. Default 42.
    """

    def __init__(
        self,
        data_path: str | None = None,
        vocab_path: str | None = None,
        config_path: str | None = None,
        n_cells: int = DEFAULT_N_CELLS,
        sample_seed: int = 42,
        max_retries: int = 3,
    ) -> None:
        self._data_path = data_path or DEFAULT_DATA_PATH
        self._vocab_path = vocab_path or DEFAULT_VOCAB_PATH
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._n_cells = n_cells
        self._sample_seed = sample_seed
        self._max_retries = max_retries

        # Lazily loaded on first evaluate call.
        self._data_loader: DataLoader | None = None
        self._vocab: GeneVocabulary | None = None
        self._evaluator: ADNCEvaluator | None = None

        # Fallback guardrail state.
        self._last_valid_selection: list[str] | None = None
        self._consecutive_failures: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API (OpenEvolve interface)
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_stage1(self, select_genes_func) -> EvaluationResult:
        """
        Evaluate the evolved gene selector function (primary stage).

        OpenEvolve calls this with the evolved select_genes function each
        generation.  Returns an EvaluationResult whose metrics['primary'] is
        the balanced accuracy for ADNC classification, and whose artifacts
        contain detailed feedback for the LLM.

        Args:
            select_genes_func: The evolved callable with signature
                select_genes(gene_vocabulary: list, all_genes: list,
                             previous_results: dict | None) -> list[str]

        Returns:
            EvaluationResult with:
                metrics = {
                    'primary': float,           # balanced accuracy (main fitness)
                    'balanced_accuracy': float,
                    'macro_f1': float,
                    'per_class_f1': dict,       # class label → F1
                    'n_retained': int,          # genes with non-zero coef
                }
                artifacts = {
                    'retained_genes': list[str],
                    'coefficients': dict[str, float],
                    'suggestions': list[str],
                    'selected_genes': list[str],
                }
        """
        try:
            self._ensure_loaded()

            gene_vocab_list = self._vocab.in_vocab
            all_genes_list = list(self._data_loader.adata.var_names)

            # ── Call the evolved function ─────────────────────────────────────
            selected_genes = select_genes_func(gene_vocab_list, all_genes_list)

            # ── Validate output (with fallback guardrail) ─────────────────────
            validation_error = self._validate_selection(selected_genes, all_genes_list)
            if validation_error:
                self._consecutive_failures += 1
                if (
                    self._consecutive_failures >= self._max_retries
                    and self._last_valid_selection is not None
                ):
                    logger.warning(
                        "Validation failed %d consecutive time(s); substituting "
                        "last valid selection.",
                        self._consecutive_failures,
                    )
                    selected_genes = self._last_valid_selection
                    self._consecutive_failures = 0
                    # Fall through to pipeline with the fallback selection.
                else:
                    return EvaluationResult(
                        metrics={"primary": 0.0},
                        artifacts={"error": True, "error_message": validation_error},
                    )
            else:
                self._consecutive_failures = 0
                self._last_valid_selection = selected_genes

            # ── Run pipeline ──────────────────────────────────────────────────
            sampler = CellSampler(
                self._data_loader, selected_genes,
                seed=self._sample_seed, target="adnc",
            )
            X, y = sampler.sample(self._n_cells)
            eval_result = self._evaluator.evaluate(X, y)

            # ── Build LLM feedback artifacts ──────────────────────────────────
            suggestions = _build_suggestions(eval_result, selected_genes)

            return EvaluationResult(
                metrics={
                    "primary": eval_result.aggregate_score,
                    "balanced_accuracy": eval_result.balanced_accuracy,
                    "macro_f1": eval_result.macro_f1,
                    "per_class_f1": eval_result.per_class_f1,
                    "n_retained": eval_result.n_retained,
                },
                artifacts={
                    "retained_genes": eval_result.retained_genes,
                    "coefficients": eval_result.coefficients,
                    "suggestions": suggestions,
                    "selected_genes": selected_genes,
                },
            )

        except Exception as exc:
            err_msg = f"Evaluation failed: {exc}\n{traceback.format_exc()}"
            logger.error(err_msg)
            return EvaluationResult(
                metrics={"primary": 0.0},
                artifacts={"error": True, "error_message": err_msg},
            )

    def evaluate_stage2(self, select_genes_func) -> EvaluationResult:
        """
        Optional Stage 2 evaluation (mirrors Stage 1 for this project).

        Args:
            select_genes_func: Same evolved callable passed to evaluate_stage1.

        Returns:
            EvaluationResult identical to evaluate_stage1 output.
        """
        return self.evaluate_stage1(select_genes_func)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """
        Lazy-initialize DataLoader, GeneVocabulary, and ADNCEvaluator.

        Called before each evaluate_stage1 invocation.  Loads resources only
        once; subsequent calls are no-ops.
        """
        if self._data_loader is not None:
            return

        logger.info("Loading DataLoader from %s", self._data_path)
        self._data_loader = DataLoader(h5ad_path=self._data_path)

        logger.info("Loading GeneVocabulary from %s", self._vocab_path)
        self._vocab = GeneVocabulary(
            vocab_path=self._vocab_path,
            adata_var_names=list(self._data_loader.adata.var_names),
        )

        logger.info("Loading ADNCEvaluator from %s", self._config_path)
        self._evaluator = ADNCEvaluator(config_path=self._config_path)

    def _validate_selection(
        self, selected_genes: object, all_genes: list[str]
    ) -> str | None:
        """
        Validate the gene selection returned by the evolved function.

        Checks that the selection is a list of exactly 200 unique gene symbols
        that are all present in the dataset.

        Args:
            selected_genes: Value returned by select_genes_func (any type).
            all_genes: Full list of gene symbols available in the dataset.

        Returns:
            Error message string if invalid, or None if valid.
        """
        if not isinstance(selected_genes, list):
            return (
                f"select_genes() must return a list, got {type(selected_genes).__name__}"
            )

        if len(selected_genes) != 200:
            return (
                f"select_genes() must return exactly 200 genes, got {len(selected_genes)}"
            )

        if len(set(selected_genes)) != 200:
            n_dupes = 200 - len(set(selected_genes))
            return f"select_genes() returned {n_dupes} duplicate gene(s)"

        all_genes_set = set(all_genes)
        invalid = [g for g in selected_genes if g not in all_genes_set]
        if invalid:
            examples = invalid[:5]
            return (
                f"select_genes() returned {len(invalid)} gene(s) not in dataset. "
                f"Examples: {examples}"
            )

        return None


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_suggestions(eval_result, selected_genes: list[str]) -> list[str]:
    """
    Build human-readable suggestions from ADNC classification evaluation
    results for LLM feedback.

    Provides targeted advice based on balanced accuracy, per-class F1 scores,
    gene retention rate, and top-coefficient genes.

    Args:
        eval_result: EvalResult from ADNCEvaluator.evaluate().
        selected_genes: The 200-gene list that was evaluated.

    Returns:
        List of suggestion strings to include in EvaluationResult artifacts.
    """
    suggestions = []
    n_selected = len(selected_genes)
    n_retained = eval_result.n_retained
    retention_pct = round(100 * n_retained / n_selected) if n_selected else 0

    suggestions.append(
        f"Logistic regression retained {n_retained}/{n_selected} genes "
        f"({retention_pct}%). The {n_selected - n_retained} zeroed genes "
        f"contribute nothing to ADNC classification and should be replaced."
    )

    if retention_pct < 10:
        suggestions.append(
            "Very few genes retained (<10%). Consider selecting genes from "
            "established AD pathways (amyloid, tau, neuroinflammation, synaptic)."
        )
    elif retention_pct > 50:
        suggestions.append(
            "Many genes retained (>50%). Try replacing the weakest genes with "
            "more targeted ADNC-discriminating markers."
        )

    # Highlight worst-performing ADNC class to guide selection.
    if eval_result.per_class_f1:
        worst_class = min(eval_result.per_class_f1, key=eval_result.per_class_f1.get)
        best_class = max(eval_result.per_class_f1, key=eval_result.per_class_f1.get)
        label_map = {"0": "Not AD", "1": "Low", "2": "Intermediate", "3": "High"}
        worst_name = label_map.get(worst_class, worst_class)
        best_name = label_map.get(best_class, best_class)
        suggestions.append(
            f"Per-class F1 — best: {best_name} "
            f"(F1={eval_result.per_class_f1[best_class]:.3f}), "
            f"worst: {worst_name} "
            f"(F1={eval_result.per_class_f1[worst_class]:.3f}). "
            f"Add genes that specifically discriminate {worst_name} from other classes."
        )

    # Suggest doubling down on top-coefficient gene pathways.
    if eval_result.coefficients:
        top_genes = sorted(
            eval_result.coefficients,
            key=eval_result.coefficients.get,
            reverse=True,
        )[:5]
        suggestions.append(
            f"Top 5 genes by logistic coefficient magnitude: {top_genes}. "
            f"Consider adding pathway-related genes for these."
        )

    # Identify zeroed genes.
    zeroed = [g for g in selected_genes if g not in set(eval_result.retained_genes)]
    if zeroed:
        examples = zeroed[:10]
        suggestions.append(
            f"Zeroed genes (replace these): {examples}"
            + (f" ... and {len(zeroed) - 10} more" if len(zeroed) > 10 else "")
        )

    return suggestions
