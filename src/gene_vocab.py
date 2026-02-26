"""
gene_vocab.py — Gene vocabulary loading, partitioning, and subset sampling.

Loads a curated gene vocabulary from a text file, intersects it against the
full gene space of an AnnData object, and supports reproducible random sampling
of a mixed in-vocab / out-of-vocab gene subset.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default path to the curated gene vocabulary.
DEFAULT_VOCAB_PATH = str(
    Path(__file__).parent.parent / "config" / "gene_vocabulary.txt"
)


class GeneVocabulary:
    """
    Gene vocabulary with in-vocab / out-of-vocab partitioning and subset sampling.

    After construction, `in_vocab` contains genes that are both in the curated
    vocabulary AND present in the AnnData gene space. `out_vocab` contains
    genes that are in the AnnData gene space but NOT in the vocabulary.

    Args:
        vocab_path: Path to a plain-text file with one gene symbol per line.
                    Defaults to DEFAULT_VOCAB_PATH.
        adata_var_names: List (or array-like) of all gene names from adata.var_names.
    """

    def __init__(
        self,
        vocab_path: str = DEFAULT_VOCAB_PATH,
        adata_var_names: list = None,
    ):
        if adata_var_names is None:
            raise ValueError("adata_var_names must be provided.")

        self._vocab_path = str(vocab_path)
        self._raw_vocab = self._load_vocab(self._vocab_path)

        # Build case-normalised lookup for matching.
        adata_genes = list(adata_var_names)
        adata_upper = {g.upper(): g for g in adata_genes}  # upper → original

        matched = set()
        self._missing_from_adata: list[str] = []

        for gene in self._raw_vocab:
            key = gene.upper()
            if key in adata_upper:
                matched.add(adata_upper[key])
            else:
                self._missing_from_adata.append(gene)

        missing_rate = len(self._missing_from_adata) / len(self._raw_vocab) if self._raw_vocab else 0.0
        if missing_rate > 0.05:
            logger.warning(
                "%.1f%% of vocabulary genes (%d/%d) not found in adata. "
                "Check gene symbol conventions.",
                missing_rate * 100,
                len(self._missing_from_adata),
                len(self._raw_vocab),
            )

        self.in_vocab: list[str] = sorted(matched)
        self.out_vocab: list[str] = sorted(set(adata_genes) - matched)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_subset(
        self,
        n_in: int = 100,
        n_out: int = 100,
        seed: int = None,
    ) -> list[str]:
        """
        Return a mixed gene subset: n_in from in_vocab + n_out from out_vocab.

        Args:
            n_in: Number of in-vocab genes to sample. Default 100.
            n_out: Number of out-of-vocab genes to sample. Default 100.
            seed: Random seed for reproducibility. None means non-deterministic.

        Returns:
            List of (n_in + n_out) unique gene symbols.

        Raises:
            ValueError: If in_vocab or out_vocab are too small to satisfy the request.
        """
        if len(self.in_vocab) < n_in:
            raise ValueError(
                f"Requested {n_in} in-vocab genes but only {len(self.in_vocab)} available."
            )
        if len(self.out_vocab) < n_out:
            raise ValueError(
                f"Requested {n_out} out-of-vocab genes but only {len(self.out_vocab)} available."
            )

        rng = np.random.default_rng(seed)
        sampled_in = rng.choice(self.in_vocab, size=n_in, replace=False).tolist()
        sampled_out = rng.choice(self.out_vocab, size=n_out, replace=False).tolist()
        return sampled_in + sampled_out

    def validate(self) -> dict:
        """
        Return summary statistics about vocabulary coverage.

        Returns:
            Dict with keys:
              - vocab_size: total genes in the vocabulary file
              - in_vocab_count: genes matched in adata
              - out_vocab_count: genes in adata but not in vocab
              - missing_from_adata: genes in vocab but not in adata
              - missing_from_adata_count: count of above
              - coverage_pct: percentage of vocab genes found in adata
        """
        total = len(self._raw_vocab)
        found = len(self.in_vocab)
        coverage = (found / total * 100) if total > 0 else 0.0
        return {
            "vocab_size": total,
            "in_vocab_count": found,
            "out_vocab_count": len(self.out_vocab),
            "missing_from_adata": self._missing_from_adata,
            "missing_from_adata_count": len(self._missing_from_adata),
            "coverage_pct": round(coverage, 2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_vocab(path: str) -> list[str]:
        """
        Load gene symbols from a plain-text file (one per line).

        Blank lines and lines starting with '#' are skipped.

        Args:
            path: Path to the vocabulary file.

        Returns:
            List of stripped gene symbol strings.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        genes = []
        with open(p) as f:
            for line in f:
                gene = line.strip()
                if gene and not gene.startswith("#"):
                    genes.append(gene)
        return genes
