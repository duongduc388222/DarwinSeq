"""
evaluator.py — Fixed-parameter LASSO evaluator for the SEA-AD DREAM Challenge.

Trains one Lasso regression model per pathology target on gene-expression data,
scores predictions using Pearson correlation, and returns which genes carry
non-zero coefficients.

Usage example:
    evaluator = LASSOEvaluator()
    result = evaluator.evaluate(X, y)
    print(result.aggregate_score, result.retained_genes)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = str(
    Path(__file__).parent.parent / "config" / "lasso_params.json"
)


@dataclass
class EvalResult:
    """
    Output of a single LASSOEvaluator.evaluate() call.

    Attributes:
        scores: Per-target Pearson r values keyed by target column name.
                Targets skipped due to all-NaN y are absent from this dict.
        aggregate_score: Mean Pearson r across all evaluated targets.
        retained_genes: Sorted list of genes with |coef| > coef_threshold
                        in at least one target model (union across targets).
        coefficients: Gene → mean absolute coefficient across all target models
                      in which the gene appeared. Covers all retained_genes.
        n_retained: len(retained_genes).
    """

    scores: dict = field(default_factory=dict)
    aggregate_score: float = 0.0
    retained_genes: list = field(default_factory=list)
    coefficients: dict = field(default_factory=dict)
    n_retained: int = 0


class LASSOEvaluator:
    """
    LASSO-based evaluator for gene-expression → pathology-target prediction.

    Fits one sklearn Lasso per pathology target using fixed hyperparameters
    loaded from a JSON config file. Features are StandardScaler-normalised
    before fitting. Pearson correlation is computed on the training set (the
    same data used for fitting) to produce a signal for evolutionary search.

    Args:
        config_path: Path to the JSON file with LASSO hyperparameters.
                     Defaults to DEFAULT_CONFIG_PATH.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        config_path = str(config_path)
        if not Path(config_path).exists():
            raise FileNotFoundError(f"LASSO config not found: {config_path}")
        with open(config_path) as fh:
            self._params = json.load(fh)

        self._alpha: float = float(self._params["alpha"])
        self._fit_intercept: bool = bool(self._params["fit_intercept"])
        self._max_iter: int = int(self._params["max_iter"])
        self._random_state: int = int(self._params["random_state"])
        self._coef_threshold: float = float(self._params["coef_threshold"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> EvalResult:
        """
        Train LASSO on X and y, score predictions, and return retained genes.

        Fits one Lasso per column in y. Each model is trained on the subset of
        rows where y_target is not NaN. Pearson r is computed on the same
        training rows. Retained genes are the union of non-zero coefficient
        genes across all fitted models.

        Args:
            X: Feature matrix, shape (n_samples, n_genes). Index must align
               with y's index.
            y: Target matrix, shape (n_samples, n_targets). Columns are
               pathology target names.

        Returns:
            EvalResult with per-target scores, aggregate score, retained
            genes, and coefficient magnitudes.

        Raises:
            ValueError: If X and y have different numbers of rows.
        """
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same number of rows, "
                f"got X={len(X)}, y={len(y)}."
            )
        if X.empty or y.empty:
            return EvalResult()

        gene_names = list(X.columns)
        X_scaled, _ = self._preprocess(X, y)

        all_scores: dict[str, float] = {}
        # gene → list of abs coef values across targets
        gene_coef_accum: dict[str, list[float]] = {}

        for target_col in y.columns:
            y_target = y[target_col]
            valid_mask = y_target.notna()
            n_valid = valid_mask.sum()

            if n_valid == 0:
                logger.warning("Target '%s' has no non-NaN values; skipping.", target_col)
                continue
            if n_valid < 2:
                logger.warning(
                    "Target '%s' has only %d non-NaN sample(s); skipping.",
                    target_col, n_valid,
                )
                continue

            X_fit = X_scaled[valid_mask.values]
            y_fit = y_target[valid_mask].values.astype(float)

            r, coef_dict = self._train_and_score(X_fit, y_fit, gene_names)
            all_scores[target_col] = r

            for gene, coef_abs in coef_dict.items():
                gene_coef_accum.setdefault(gene, []).append(coef_abs)

        # Aggregate score: mean Pearson r across all evaluated targets.
        if all_scores:
            aggregate = float(np.mean(list(all_scores.values())))
        else:
            aggregate = 0.0

        # Retained genes: union of non-zero-coef genes, sorted for determinism.
        retained_genes = sorted(gene_coef_accum.keys())
        coefficients = {
            gene: float(np.mean(vals))
            for gene, vals in gene_coef_accum.items()
        }

        return EvalResult(
            scores=all_scores,
            aggregate_score=aggregate,
            retained_genes=retained_genes,
            coefficients=coefficients,
            n_retained=len(retained_genes),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Scale feature matrix X with StandardScaler.

        Fit the scaler on all rows of X (NaN handling in y is done
        per-target inside evaluate()). The scaler is re-fit fresh for
        each evaluate() call so results are reproducible given the same X.

        Args:
            X: Feature matrix (n_samples, n_genes).
            y: Target matrix (passed through unchanged).

        Returns:
            Tuple of (X_scaled as np.ndarray, y unchanged as pd.DataFrame).
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values.astype(float))
        return X_scaled, y

    def _train_and_score(
        self,
        X_scaled: np.ndarray,
        y_target: np.ndarray,
        gene_names: list[str],
    ) -> tuple[float, dict[str, float]]:
        """
        Fit a single Lasso model and return Pearson r + non-zero coef genes.

        Args:
            X_scaled: Scaled feature matrix (n_samples, n_genes).
            y_target: 1-D target array (n_samples,), no NaNs.
            gene_names: Gene names matching columns of X_scaled.

        Returns:
            Tuple of:
              - pearson_r: float, Pearson correlation between y_target and
                           in-sample predictions. NaN if variance is zero.
              - coef_dict: {gene: abs(coef)} for genes with |coef| > threshold.
        """
        model = Lasso(
            alpha=self._alpha,
            fit_intercept=self._fit_intercept,
            max_iter=self._max_iter,
        )
        model.fit(X_scaled, y_target)

        y_pred = model.predict(X_scaled)

        # Pearson r: returns NaN if either array has zero variance.
        if np.std(y_target) == 0 or np.std(y_pred) == 0:
            r = float("nan")
        else:
            r, _ = pearsonr(y_target, y_pred)
            r = float(r)

        # Extract non-zero coefficients.
        coef_dict: dict[str, float] = {}
        for gene, coef in zip(gene_names, model.coef_):
            if abs(coef) > self._coef_threshold:
                coef_dict[gene] = abs(float(coef))

        return r, coef_dict
