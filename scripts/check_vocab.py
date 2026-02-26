"""
check_vocab.py — Load the gene vocabulary against actual AnnData and print stats.

Usage:
    python scripts/check_vocab.py
    python scripts/check_vocab.py --vocab_path config/gene_vocabulary.txt
    python scripts/check_vocab.py --data_path /path/to/data.h5ad

Prints vocabulary coverage statistics without modifying any files.
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DEFAULT_DATA_PATH, safe_load
from src.gene_vocab import DEFAULT_VOCAB_PATH, GeneVocabulary


def main():
    """Load vocab and adata, then print overlap statistics."""
    parser = argparse.ArgumentParser(description="Check gene vocabulary coverage against AnnData.")
    parser.add_argument(
        "--vocab_path",
        default=DEFAULT_VOCAB_PATH,
        help=f"Path to gene vocabulary file (default: {DEFAULT_VOCAB_PATH})",
    )
    parser.add_argument(
        "--data_path",
        default=DEFAULT_DATA_PATH,
        help=f"Path to .h5ad file (default: {DEFAULT_DATA_PATH})",
    )
    args = parser.parse_args()

    print(f"Vocab file : {args.vocab_path}")
    print(f"Data file  : {args.data_path}")
    print()

    adata = safe_load(h5ad_path=args.data_path)

    gv = GeneVocabulary(
        vocab_path=args.vocab_path,
        adata_var_names=list(adata.var_names),
    )

    stats = gv.validate()

    print("=== Gene Vocabulary Coverage ===")
    print(f"  Vocabulary size       : {stats['vocab_size']:,}")
    print(f"  In-vocab (in adata)   : {stats['in_vocab_count']:,}")
    print(f"  Out-of-vocab (in adata, not in vocab) : {stats['out_vocab_count']:,}")
    print(f"  Missing from adata    : {stats['missing_from_adata_count']:,}")
    print(f"  Coverage              : {stats['coverage_pct']:.1f}%")
    print()

    if stats["missing_from_adata_count"] > 0:
        print("First 20 vocab genes missing from adata:")
        for g in stats["missing_from_adata"][:20]:
            print(f"  {g}")
        if stats["missing_from_adata_count"] > 20:
            print(f"  ... and {stats['missing_from_adata_count'] - 20} more")
        print()

    # Demonstrate sample_subset
    subset = gv.sample_subset(n_in=100, n_out=100, seed=42)
    print(f"Sample subset (seed=42): {len(subset)} genes "
          f"({sum(g in set(gv.in_vocab) for g in subset)} in-vocab, "
          f"{sum(g in set(gv.out_vocab) for g in subset)} out-of-vocab)")
    print("First 5:", subset[:5])


if __name__ == "__main__":
    main()
