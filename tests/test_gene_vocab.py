"""
test_gene_vocab.py — Unit tests for src/gene_vocab.py.

All tests use synthetic gene lists so no real h5ad file is required.
"""

import pytest

from src.gene_vocab import GeneVocabulary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_vocab_file(tmp_path, genes: list[str]) -> str:
    """Write a list of gene symbols to a temp vocab file and return its path."""
    p = tmp_path / "vocab.txt"
    p.write_text("\n".join(genes) + "\n")
    return str(p)


@pytest.fixture
def vocab_genes():
    """200 in-vocab genes: GENE_V_0 … GENE_V_199."""
    return [f"GENE_V_{i}" for i in range(200)]


@pytest.fixture
def adata_genes(vocab_genes):
    """
    Full adata gene space: the 200 vocab genes + 500 extra out-of-vocab genes.
    Total 700 genes.
    """
    extra = [f"GENE_X_{i}" for i in range(500)]
    return vocab_genes + extra


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_in_vocab_count(tmp_path, vocab_genes, adata_genes):
    """All vocab genes present in adata should appear in in_vocab."""
    path = _make_vocab_file(tmp_path, vocab_genes)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    assert len(gv.in_vocab) == 200


def test_out_vocab_count(tmp_path, vocab_genes, adata_genes):
    """Genes in adata but not in vocab should appear in out_vocab."""
    path = _make_vocab_file(tmp_path, vocab_genes)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    assert len(gv.out_vocab) == 500


def test_no_overlap_between_partitions(tmp_path, vocab_genes, adata_genes):
    """in_vocab and out_vocab must be disjoint."""
    path = _make_vocab_file(tmp_path, vocab_genes)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    assert set(gv.in_vocab).isdisjoint(set(gv.out_vocab))


def test_union_covers_adata(tmp_path, vocab_genes, adata_genes):
    """Union of in_vocab and out_vocab should equal the full adata gene space."""
    path = _make_vocab_file(tmp_path, vocab_genes)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    assert set(gv.in_vocab) | set(gv.out_vocab) == set(adata_genes)


# ---------------------------------------------------------------------------
# sample_subset
# ---------------------------------------------------------------------------

def test_sample_subset_length(tmp_path, vocab_genes, adata_genes):
    """sample_subset() should return exactly n_in + n_out genes."""
    path = _make_vocab_file(tmp_path, vocab_genes)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    subset = gv.sample_subset(n_in=100, n_out=100, seed=42)
    assert len(subset) == 200


def test_sample_subset_all_unique(tmp_path, vocab_genes, adata_genes):
    """sample_subset() should contain no duplicate genes."""
    path = _make_vocab_file(tmp_path, vocab_genes)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    subset = gv.sample_subset(n_in=100, n_out=100, seed=42)
    assert len(subset) == len(set(subset))


def test_sample_subset_in_out_split(tmp_path, vocab_genes, adata_genes):
    """First n_in genes should be in-vocab; next n_out should be out-of-vocab."""
    path = _make_vocab_file(tmp_path, vocab_genes)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    subset = gv.sample_subset(n_in=100, n_out=100, seed=42)

    in_set = set(gv.in_vocab)
    out_set = set(gv.out_vocab)

    sampled_in = subset[:100]
    sampled_out = subset[100:]

    assert all(g in in_set for g in sampled_in), "First 100 should be in-vocab"
    assert all(g in out_set for g in sampled_out), "Last 100 should be out-of-vocab"


def test_sample_subset_reproducible(tmp_path, vocab_genes, adata_genes):
    """Same seed should produce identical subsets."""
    path = _make_vocab_file(tmp_path, vocab_genes)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    s1 = gv.sample_subset(seed=7)
    s2 = gv.sample_subset(seed=7)
    assert s1 == s2


def test_sample_subset_different_seeds_differ(tmp_path, vocab_genes, adata_genes):
    """Different seeds should (almost certainly) produce different subsets."""
    path = _make_vocab_file(tmp_path, vocab_genes)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    s1 = gv.sample_subset(seed=1)
    s2 = gv.sample_subset(seed=2)
    assert s1 != s2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_too_few_in_vocab_raises(tmp_path, adata_genes):
    """Requesting more in-vocab genes than available should raise ValueError."""
    small_vocab = [f"GENE_V_{i}" for i in range(5)]  # only 5 vocab genes
    path = _make_vocab_file(tmp_path, small_vocab)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    with pytest.raises(ValueError, match="in-vocab"):
        gv.sample_subset(n_in=100, n_out=100)


def test_too_few_out_vocab_raises(tmp_path):
    """Requesting more out-of-vocab genes than available should raise ValueError."""
    # Make a vocab that covers everything in adata → no out-of-vocab genes
    adata_genes = [f"GENE_{i}" for i in range(50)]
    path = _make_vocab_file(tmp_path, adata_genes)  # vocab == adata
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    with pytest.raises(ValueError, match="out-of-vocab"):
        gv.sample_subset(n_in=10, n_out=1)


def test_vocab_missing_from_adata_tracked(tmp_path, adata_genes):
    """Genes in vocab but absent from adata should be recorded in missing_from_adata."""
    extra_vocab = [f"GHOST_GENE_{i}" for i in range(10)]
    all_vocab = [f"GENE_V_{i}" for i in range(200)] + extra_vocab
    path = _make_vocab_file(tmp_path, all_vocab)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    stats = gv.validate()
    assert stats["missing_from_adata_count"] == 10
    assert all(g in stats["missing_from_adata"] for g in extra_vocab)


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

def test_validate_returns_correct_stats(tmp_path, vocab_genes, adata_genes):
    """validate() should return accurate coverage statistics."""
    path = _make_vocab_file(tmp_path, vocab_genes)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata_genes)
    stats = gv.validate()
    assert stats["vocab_size"] == 200
    assert stats["in_vocab_count"] == 200
    assert stats["out_vocab_count"] == 500
    assert stats["missing_from_adata_count"] == 0
    assert stats["coverage_pct"] == 100.0


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def test_missing_vocab_file_raises(tmp_path, adata_genes):
    """FileNotFoundError should be raised if vocab file does not exist."""
    with pytest.raises(FileNotFoundError):
        GeneVocabulary(
            vocab_path=str(tmp_path / "nonexistent.txt"),
            adata_var_names=adata_genes,
        )


def test_blank_lines_and_comments_ignored(tmp_path, adata_genes):
    """Blank lines and #-comment lines in vocab file should be skipped."""
    p = tmp_path / "vocab.txt"
    p.write_text("# comment\n\nGENE_V_0\n\nGENE_V_1\n")
    gv = GeneVocabulary(vocab_path=str(p), adata_var_names=adata_genes)
    assert len(gv.in_vocab) == 2


def test_case_insensitive_matching(tmp_path):
    """Vocab genes should match adata genes case-insensitively."""
    vocab = ["tp53", "BRCA1", "Myc"]
    adata = ["TP53", "BRCA1", "MYC", "KRAS"]
    path = _make_vocab_file(tmp_path, vocab)
    gv = GeneVocabulary(vocab_path=path, adata_var_names=adata)
    assert len(gv.in_vocab) == 3
    assert len(gv.out_vocab) == 1
    assert "KRAS" in gv.out_vocab


def test_no_adata_var_names_raises():
    """Omitting adata_var_names should raise ValueError."""
    with pytest.raises(ValueError):
        GeneVocabulary(adata_var_names=None)
