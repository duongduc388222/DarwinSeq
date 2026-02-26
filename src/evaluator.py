"""
evaluator.py — ADNC classification evaluator for the SEA-AD DREAM Challenge.

Trains a single L1-penalised logistic regression model on gene-expression data
to predict ADNC class (Not AD=0 / Low=1 / Intermediate=2 / High=3), scores
predictions using balanced accuracy, and returns which genes carry non-zero
coefficients across the one-vs-rest (OvR) binary classifiers.

Usage example:
    evaluator = ADNCEvaluator()
    result = evaluator.evaluate(X, y)
    print(result.balanced_accuracy, result.retained_genes)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

# sklearn ≥1.8 deprecates 'penalty' in favour of 'l1_ratio'.
# sklearn ≥1.8 also requires OneVsRestClassifier for liblinear multiclass.
_SKLEARN_GE_18 = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = str(
    Path(__file__).parent.parent / "config" / "model_params.json"
)


@dataclass
class EvalResult:
    """
    Output of a single ADNCEvaluator.evaluate() call.

    Attributes:
        balanced_accuracy: Primary metric — sklearn balanced_accuracy_score
                           on the training set.
        macro_f1: Macro-averaged F1 across all ADNC classes seen in the sample.
        aggregate_score: Alias for balanced_accuracy. Kept so the OpenEvolve
                         adapter and run_baseline.py can use a stable key.
        retained_genes: Sorted list of genes whose sum-of-|coef| across all
                        OvR classifiers exceeds coef_threshold.
        coefficients: Gene → sum of |coef| across classes. Covers all
                      retained_genes.
        n_retained: len(retained_genes).
        per_class_f1: Class label (str) → per-class F1 score.
    """

    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    aggregate_score: float = 0.0
    retained_genes: list = field(default_factory=list)
    coefficients: dict = field(default_factory=dict)
    n_retained: int = 0
    per_class_f1: dict = field(default_factory=dict)


class ADNCEvaluator:
    """
    L1 logistic regression evaluator for ADNC 4-class prediction.

    Fits a single LogisticRegression(penalty='l1', solver='liblinear') on
    gene-expression features to predict ADNC class. StandardScaler
    normalisation is applied before fitting. Balanced accuracy is the primary
    fitness signal returned to the gene-selection loop.

    Args:
        config_path: Path to the JSON config file (model_params.json).
                     Defaults to DEFAULT_CONFIG_PATH.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        config_path = str(config_path)
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        with open(config_path) as fh:
            self._params = json.load(fh)

        self._C: float = float(self._params["C"])
        self._l1_ratio: float = float(self._params["l1_ratio"])
        self._solver: str = str(self._params["solver"])
        self._class_weight = self._params["class_weight"]
        self._max_iter: int = int(self._params["max_iter"])
        self._random_state: int = int(self._params["random_state"])
        self._coef_threshold: float = float(self._params["coef_threshold"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> EvalResult:
        """
        Train L1 logistic regression on X predicting ADNC class in y.

        y must have a single column (named "ADNC" by convention) containing
        integer-encoded ADNC labels (0=Not AD, 1=Low, 2=Intermediate, 3=High).
        Rows where y is NaN are silently dropped before fitting.

        Args:
            X: Feature matrix, shape (n_cells, n_genes). Index must align
               with y's index.
            y: Single-column DataFrame with ADNC integer labels (or NaN).

        Returns:
            EvalResult with balanced_accuracy, macro_f1, retained_genes,
            coefficients, per_class_f1, and n_retained.

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

        # Drop NaN-labelled rows.
        target_col = y.columns[0]
        y_col = y[target_col]
        valid_mask = y_col.notna()
        n_valid = int(valid_mask.sum())

        if n_valid < 10:
            logger.warning(
                "Fewer than 10 labelled cells (%d); returning zero result.", n_valid
            )
            return EvalResult()

        y_labels = y_col[valid_mask].values.astype(int)

        if len(np.unique(y_labels)) < 2:
            logger.warning(
                "Only one ADNC class present in sample; returning zero result."
            )
            return EvalResult()

        gene_names = list(X.columns)
        X_scaled, _ = self._preprocess(X, y)
        X_fit = X_scaled[valid_mask.values]

        bal_acc, macro_f1, coef_dict, per_class_f1 = self._train_and_score(
            X_fit, y_labels, gene_names
        )

        retained_genes = sorted(coef_dict.keys())
        return EvalResult(
            balanced_accuracy=bal_acc,
            macro_f1=macro_f1,
            aggregate_score=bal_acc,
            retained_genes=retained_genes,
            coefficients=coef_dict,
            n_retained=len(retained_genes),
            per_class_f1=per_class_f1,
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

        The scaler is re-fit from scratch on each evaluate() call so results
        are fully reproducible given the same X, regardless of call order.

        Args:
            X: Feature matrix (n_cells, n_genes).
            y: Target DataFrame (passed through unchanged).

        Returns:
            Tuple of (X_scaled as np.ndarray, y unchanged as pd.DataFrame).
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values.astype(float))
        return X_scaled, y

    def _train_and_score(
        self,
        X_scaled: np.ndarray,
        y_labels: np.ndarray,
        gene_names: list[str],
    ) -> tuple[float, float, dict[str, float], dict[str, float]]:
        """
        Fit a single L1 logistic regression model and return classification
        metrics plus gene importance coefficients.

        Args:
            X_scaled: Scaled feature matrix (n_valid_cells, n_genes).
            y_labels: 1-D integer label array (n_valid_cells,), no NaNs.
            gene_names: Gene names matching columns of X_scaled.

        Returns:
            Tuple of:
              - balanced_accuracy: float, in-sample balanced accuracy.
              - macro_f1: float, macro-averaged F1.
              - coef_dict: {gene: sum_abs_coef} for genes above threshold.
              - per_class_f1: {class_label_str: f1_score} for each class.
        """
        # sklearn ≥1.8: 'penalty' is deprecated; use l1_ratio=1 for L1.
        #              liblinear no longer handles multiclass natively →
        #              wrap in OneVsRestClassifier.
        # sklearn <1.8: use penalty='l1' directly; OvR wrapper still works.
        common_kwargs = dict(
            solver=self._solver,
            C=self._C,
            class_weight=self._class_weight,
            max_iter=self._max_iter,
            random_state=self._random_state,
        )
        if _SKLEARN_GE_18:
            base_clf = LogisticRegression(l1_ratio=self._l1_ratio, **common_kwargs)
        else:
            base_clf = LogisticRegression(penalty="l1", **common_kwargs)
        model = OneVsRestClassifier(base_clf)
        model.fit(X_scaled, y_labels)
        y_pred = model.predict(X_scaled)

        bal_acc = float(balanced_accuracy_score(y_labels, y_pred))
        macro_f1 = float(
            f1_score(y_labels, y_pred, average="macro", zero_division=0)
        )
        per_class_scores = f1_score(
            y_labels, y_pred, average=None, zero_division=0
        )
        per_class_f1 = {
            str(cls): float(score)
            for cls, score in zip(model.classes_, per_class_scores)
        }

        # Gene importance: stack each binary classifier's |coef| then sum.
        # Each estimator.coef_ has shape (1, n_genes) for liblinear.
        coef_matrix = np.vstack([est.coef_ for est in model.estimators_])
        importance = np.sum(np.abs(coef_matrix), axis=0)
        coef_dict: dict[str, float] = {
            gene: float(importance[i])
            for i, gene in enumerate(gene_names)
            if importance[i] > self._coef_threshold
        }

        return bal_acc, macro_f1, coef_dict, per_class_f1
