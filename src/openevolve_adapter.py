"""
openevolve_adapter.py — Adapter connecting DarwinSeq's pipeline to OpenEvolve's
evaluator interface.

OpenEvolve calls GeneSelectorEvaluator.evaluate_stage1(select_genes_func) each
generation with the evolved function.  The adapter:
  1. Calls select_genes_func(gene_vocabulary, all_genes) → 200 genes
  2. Validates the output (count, uniqueness, presence in dataset)
  3. Samples 100 cells and runs the LASSO evaluator
  4. Returns EvaluationResult with fitness (Pearson r) and LLM feedback artifacts

The DataLoader and other heavy objects are loaded lazily on the first call so
that tests can instantiate GeneSelectorEvaluator without a real h5ad file.
"""

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
from src.evaluator import DEFAULT_CONFIG_PATH, LASSOEvaluator
from src.gene_vocab import DEFAULT_VOCAB_PATH, GeneVocabulary
from src.sampler import CellSampler

logger = logging.getLogger(__name__)

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
        config_path: Path to the LASSO JSON config. Defaults to DEFAULT_CONFIG_PATH.
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
    ) -> None:
        self._data_path = data_path or DEFAULT_DATA_PATH
        self._vocab_path = vocab_path or DEFAULT_VOCAB_PATH
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._n_cells = n_cells
        self._sample_seed = sample_seed

        # Lazily loaded on first evaluate call.
        self._data_loader: DataLoader | None = None
        self._vocab: GeneVocabulary | None = None
        self._evaluator: LASSOEvaluator | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API (OpenEvolve interface)
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_stage1(self, select_genes_func) -> EvaluationResult:
        """
        Evaluate the evolved gene selector function (primary stage).

        OpenEvolve calls this with the evolved select_genes function each
        generation.  Returns an EvaluationResult whose metrics['primary'] is
        the aggregate LASSO Pearson r score and whose artifacts contain detailed
        feedback for the LLM (retained genes, coefficients, suggestions).

        Args:
            select_genes_func: The evolved callable with signature
                select_genes(gene_vocabulary: list, all_genes: list,
                             previous_results: dict | None) -> list[str]

        Returns:
            EvaluationResult with:
                metrics = {
                    'primary': float,          # aggregate Pearson r (main fitness)
                    'per_target_scores': dict, # per-target Pearson r values
                    'n_retained': int,         # genes with non-zero LASSO coef
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

            # ── Validate output ───────────────────────────────────────────────
            validation_error = self._validate_selection(selected_genes, all_genes_list)
            if validation_error:
                return EvaluationResult(
                    metrics={"primary": 0.0},
                    artifacts={"error": True, "error_message": validation_error},
                )

            # ── Run pipeline ──────────────────────────────────────────────────
            sampler = CellSampler(
                self._data_loader, selected_genes, seed=self._sample_seed
            )
            X, y = sampler.sample(self._n_cells)
            eval_result = self._evaluator.evaluate(X, y)

            # ── Build LLM feedback artifacts ──────────────────────────────────
            suggestions = _build_suggestions(eval_result, selected_genes)

            return EvaluationResult(
                metrics={
                    "primary": eval_result.aggregate_score,
                    "per_target_scores": eval_result.scores,
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
        Lazy-initialize DataLoader, GeneVocabulary, and LASSOEvaluator.

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

        logger.info("Loading LASSOEvaluator from %s", self._config_path)
        self._evaluator = LASSOEvaluator(config_path=self._config_path)

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
    Build human-readable suggestions from LASSO evaluation for LLM feedback.

    Provides targeted advice based on how many genes were retained, which
    targets scored low, and general guidance on improving the selection.

    Args:
        eval_result: EvalResult from LASSOEvaluator.evaluate().
        selected_genes: The 200-gene list that was evaluated.

    Returns:
        List of suggestion strings to include in EvaluationResult artifacts.
    """
    suggestions = []
    n_selected = len(selected_genes)
    n_retained = eval_result.n_retained
    retention_pct = round(100 * n_retained / n_selected) if n_selected else 0

    suggestions.append(
        f"LASSO retained {n_retained}/{n_selected} genes ({retention_pct}%). "
        f"The {n_selected - n_retained} zeroed genes should be replaced."
    )

    if retention_pct < 10:
        suggestions.append(
            "Very few genes were retained (<10%). Consider selecting genes from "
            "established AD pathways (amyloid, tau, inflammation, synaptic)."
        )
    elif retention_pct > 50:
        suggestions.append(
            "Many genes were retained (>50%). The regularization may not be strong "
            "enough — try replacing weakest genes with more targeted AD markers."
        )

    # Identify best and worst targets.
    valid_scores = {k: v for k, v in eval_result.scores.items() if v == v}  # exclude NaN
    if valid_scores:
        best_target = max(valid_scores, key=valid_scores.get)
        worst_target = min(valid_scores, key=valid_scores.get)
        suggestions.append(
            f"Best target: {best_target} (r={valid_scores[best_target]:.3f}). "
            f"Worst target: {worst_target} (r={valid_scores[worst_target]:.3f})."
        )

    # Suggest doubling down on top-coefficient genes' pathways.
    if eval_result.coefficients:
        top_genes = sorted(
            eval_result.coefficients, key=eval_result.coefficients.get, reverse=True
        )[:5]
        suggestions.append(
            f"Top 5 genes by LASSO coefficient: {top_genes}. "
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
