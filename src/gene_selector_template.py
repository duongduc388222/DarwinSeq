"""
gene_selector_template.py — Initial gene selection program evolved by OpenEvolve.

OpenEvolve will iteratively rewrite the code inside the EVOLVE-BLOCK markers,
using LLM-guided mutations and LASSO evaluation feedback to improve gene selection
for Alzheimer's Disease pathology prediction.

The function signature and return contract (exactly 200 genes) must never change.
"""


def select_genes(
    gene_vocabulary: list,
    all_genes: list,
    previous_results: dict | None = None,
) -> list:
    """
    Select 200 genes for LASSO evaluation of AD pathology targets.

    OpenEvolve rewrites the code inside EVOLVE-BLOCK each generation based on
    LASSO feedback from prior runs.

    Args:
        gene_vocabulary: The curated AD-relevant gene vocabulary (~3,157 genes).
        all_genes: All gene symbols available in the SEAAD A9 dataset.
        previous_results: Optional dict from the prior generation with keys:
            - 'aggregate_score': float — mean Pearson r across 6 pathology targets
            - 'retained_genes': list[str] — genes with non-zero LASSO coefficient
            - 'coefficients': dict[str, float] — gene → mean absolute coefficient

    Returns:
        List of exactly 200 gene symbols: 100 from gene_vocabulary (in-vocab)
        and 100 from outside gene_vocabulary (out-of-vocab).
    """
    # EVOLVE-BLOCK-START

    import random

    # Initial version: uniform random selection split evenly between in-vocab
    # and out-of-vocab genes to match the baseline run distribution.
    # OpenEvolve will replace this logic with biologically-informed selection.

    random.seed(42)
    vocab_set = set(gene_vocabulary)
    out_of_vocab = [g for g in all_genes if g not in vocab_set]

    in_vocab_selected = random.sample(gene_vocabulary, 100)
    out_vocab_selected = random.sample(out_of_vocab, 100)

    return in_vocab_selected + out_vocab_selected

    # EVOLVE-BLOCK-END
