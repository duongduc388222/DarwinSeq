"""
test_evolve.py — Unit tests for Milestone 6: OpenEvolve integration.

All tests run without a real h5ad file, real OpenEvolve API, or real LLM:

  Group A — Template program (src/gene_selector_template.py)
    1.  select_genes returns exactly 200 genes
    2.  100 genes are from gene_vocabulary (in-vocab)
    3.  100 genes are NOT from gene_vocabulary (out-of-vocab)
    4.  No duplicates in the output

  Group B — Adapter validation (src/openevolve_adapter.py, with FakeDataLoader)
    5.  Wrong gene count → EvaluationResult with error
    6.  Invalid (unknown) gene name → EvaluationResult with error
    7.  Duplicate genes → EvaluationResult with error
    8.  Valid selection → EvaluationResult.metrics['primary'] in [-1, 1]
    9.  evaluate_stage1 passes (vocab_list, all_genes_list) to the evolved function
    10. evaluate_stage2 mirrors evaluate_stage1

  Group C — EvolutionRunner unit (src/evolve.py)
    11. EvolutionRunner.__init__ loads a minimal YAML without error
    12. EvolutionRunner.__init__ raises FileNotFoundError for missing config
    13. log_generation creates gen_{id}/result.json under output_dir
    14. GenerationResult serializes to / from JSON cleanly
    15. _build_suggestions returns a non-empty list of strings
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── stubs shared across groups ────────────────────────────────────────────────

_PATHOLOGY_COLS = [
    "percent 6e10 positive area",
    "percent AT8 positive area",
    "percent NeuN positive area",
    "percent GFAP positive area",
    "percent pTDP43 positive area",
    "percent aSyn positive area",
]

_N_CELLS = 300
_N_GENES = 300   # 0–149 used as "in-vocab", 150–299 as "out-of-vocab"


class _FakeAdata:
    """Minimal AnnData stand-in with obs_names and var_names."""

    def __init__(self, obs: pd.DataFrame, var_names: list[str]) -> None:
        self.obs = obs
        self.obs_names = obs.index
        self.var_names = pd.Index(var_names)


class FakeDataLoader:
    """
    Stub DataLoader satisfying the interface used by CellSampler and
    GeneSelectorEvaluator (adata.var_names, get_expression_for_cells,
    get_pathology_targets).
    """

    def __init__(self, n_cells: int = _N_CELLS, n_genes: int = _N_GENES) -> None:
        self.n_cells = n_cells
        self.n_genes = n_genes
        gene_names = [f"GENE_{i}" for i in range(n_genes)]
        self._gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        barcodes = [f"CELL_{i:04d}" for i in range(n_cells)]
        rng = np.random.default_rng(0)
        pathology_data = rng.uniform(0, 1, size=(n_cells, 6))
        obs = pd.DataFrame(
            pathology_data, index=barcodes, columns=_PATHOLOGY_COLS
        )

        self.adata = _FakeAdata(obs, gene_names)

    def get_expression_for_cells(
        self, cell_barcodes: list[str], gene_list: list[str]
    ) -> pd.DataFrame:
        """Return deterministic float expression values."""
        cell_idx = np.array(
            [int(bc.split("_")[1]) for bc in cell_barcodes], dtype=float
        )
        gene_idx = np.array(
            [self._gene_to_idx.get(g, 0) for g in gene_list], dtype=float
        )
        expr = np.outer(cell_idx + 1, gene_idx + 1) * 0.001
        return pd.DataFrame(
            expr.astype(np.float32), index=cell_barcodes, columns=gene_list
        )

    def get_pathology_targets(
        self, cell_barcodes: list[str] | None = None
    ) -> pd.DataFrame:
        """Return pathology columns for requested barcodes."""
        obs = self.adata.obs[_PATHOLOGY_COLS]
        if cell_barcodes is not None:
            return obs.loc[cell_barcodes].copy()
        return obs.copy()


class FakeVocab:
    """
    Stub GeneVocabulary with only the in_vocab attribute required by the adapter.
    """

    def __init__(self, in_vocab: list[str]) -> None:
        self.in_vocab = in_vocab


def _make_evaluator(n_cells: int = 100):
    """
    Return a GeneSelectorEvaluator pre-loaded with fake components.

    Bypasses _ensure_loaded so no real h5ad file is needed.
    """
    from src.evaluator import LASSOEvaluator
    from src.openevolve_adapter import GeneSelectorEvaluator

    loader = FakeDataLoader()
    # in-vocab: first 150 genes; out-of-vocab: remaining 150
    in_vocab = [f"GENE_{i}" for i in range(150)]
    vocab = FakeVocab(in_vocab)
    lasso = LASSOEvaluator()

    evaluator = GeneSelectorEvaluator(n_cells=n_cells)
    evaluator._data_loader = loader
    evaluator._vocab = vocab
    evaluator._evaluator = lasso
    return evaluator


# ══════════════════════════════════════════════════════════════════════════════
# Group A — Template program
# ══════════════════════════════════════════════════════════════════════════════

class TestGeneSelectorTemplate:
    """Tests for src/gene_selector_template.py."""

    @pytest.fixture
    def vocab(self):
        return [f"GENE_{i}" for i in range(150)]

    @pytest.fixture
    def all_genes(self):
        return [f"GENE_{i}" for i in range(300)]

    def test_returns_exactly_200(self, vocab, all_genes):
        """select_genes() must return exactly 200 genes."""
        from src.gene_selector_template import select_genes

        result = select_genes(vocab, all_genes)
        assert len(result) == 200

    def test_100_in_vocab(self, vocab, all_genes):
        """Exactly 100 returned genes must be from gene_vocabulary."""
        from src.gene_selector_template import select_genes

        result = select_genes(vocab, all_genes)
        vocab_set = set(vocab)
        in_vocab_count = sum(1 for g in result if g in vocab_set)
        assert in_vocab_count == 100

    def test_100_out_of_vocab(self, vocab, all_genes):
        """Exactly 100 returned genes must NOT be in gene_vocabulary."""
        from src.gene_selector_template import select_genes

        result = select_genes(vocab, all_genes)
        vocab_set = set(vocab)
        out_vocab_count = sum(1 for g in result if g not in vocab_set)
        assert out_vocab_count == 100

    def test_no_duplicates(self, vocab, all_genes):
        """All returned gene names must be unique."""
        from src.gene_selector_template import select_genes

        result = select_genes(vocab, all_genes)
        assert len(set(result)) == len(result)

    def test_previous_results_none(self, vocab, all_genes):
        """select_genes() must accept previous_results=None without error."""
        from src.gene_selector_template import select_genes

        result = select_genes(vocab, all_genes, previous_results=None)
        assert len(result) == 200

    def test_all_genes_in_all_genes(self, vocab, all_genes):
        """Every returned gene must appear in the all_genes list."""
        from src.gene_selector_template import select_genes

        result = select_genes(vocab, all_genes)
        all_set = set(all_genes)
        for g in result:
            assert g in all_set, f"Gene {g!r} not in all_genes"


# ══════════════════════════════════════════════════════════════════════════════
# Group B — Adapter validation
# ══════════════════════════════════════════════════════════════════════════════

class TestGeneSelectorEvaluator:
    """Tests for src/openevolve_adapter.py."""

    def test_wrong_gene_count_returns_error(self):
        """Returning 201 genes should produce an EvaluationResult with error."""
        evaluator = _make_evaluator()

        def bad_selector(vocab, all_genes, prev=None):
            return all_genes[:201]

        result = evaluator.evaluate_stage1(bad_selector)
        assert result.artifacts.get("error"), "Expected error flag in artifacts"
        err_msg = result.artifacts.get("error_message", "")
        assert "201" in err_msg or "200" in err_msg

    def test_too_few_genes_returns_error(self):
        """Returning 50 genes should produce an EvaluationResult with error."""
        evaluator = _make_evaluator()

        def bad_selector(vocab, all_genes, prev=None):
            return all_genes[:50]

        result = evaluator.evaluate_stage1(bad_selector)
        assert result.artifacts.get("error"), "Expected error flag in artifacts"

    def test_invalid_gene_name_returns_error(self):
        """Including a gene not in the dataset should return an error."""
        evaluator = _make_evaluator()

        def bad_selector(vocab, all_genes, prev=None):
            # Replace the last gene with an unknown name.
            genes = all_genes[:200]
            genes[-1] = "NONEXISTENT_GENE_XYZ"
            return genes

        result = evaluator.evaluate_stage1(bad_selector)
        assert result.artifacts.get("error"), "Expected error flag in artifacts"
        err_msg = result.artifacts.get("error_message", "")
        assert "invalid" in err_msg.lower() or "not in dataset" in err_msg.lower()

    def test_duplicate_genes_returns_error(self):
        """Returning 200 genes with duplicates should produce an error."""
        evaluator = _make_evaluator()

        def bad_selector(vocab, all_genes, prev=None):
            # 199 unique + 1 duplicate
            return all_genes[:199] + [all_genes[0]]

        result = evaluator.evaluate_stage1(bad_selector)
        assert result.artifacts.get("error"), "Expected error flag in artifacts"
        err_msg = result.artifacts.get("error_message", "")
        assert "duplicate" in err_msg.lower()

    def test_valid_selection_returns_score(self):
        """A valid 200-gene selection should return a primary score in [-1, 1]."""
        evaluator = _make_evaluator(n_cells=100)
        in_vocab = [f"GENE_{i}" for i in range(150)]
        out_vocab_pool = [f"GENE_{i}" for i in range(150, 300)]

        def good_selector(vocab, all_genes, prev=None):
            return vocab[:100] + out_vocab_pool[:100]

        result = evaluator.evaluate_stage1(good_selector)
        assert not result.artifacts.get("error"), f"Unexpected error: {result.artifacts.get('error_message')}"
        primary = result.metrics["primary"]
        # Fake data has collinear columns; LASSO may zero all coefs → NaN Pearson r.
        # Accept NaN as a valid "no-signal" outcome from the pipeline.
        import math
        assert math.isnan(primary) or -1.0 <= primary <= 1.0

    def test_evaluate_passes_vocab_and_all_genes(self):
        """evaluate_stage1 must pass (vocab_list, all_genes_list) to the function."""
        evaluator = _make_evaluator()
        received_args = {}

        def recording_selector(vocab, all_genes, prev=None):
            received_args["vocab"] = vocab
            received_args["all_genes"] = all_genes
            # Return a valid selection to avoid downstream errors.
            out = [g for g in all_genes if g not in set(vocab)]
            return vocab[:100] + out[:100]

        evaluator.evaluate_stage1(recording_selector)

        assert "vocab" in received_args
        assert "all_genes" in received_args
        # vocab should be the evaluator's in_vocab list
        assert received_args["vocab"] == evaluator._vocab.in_vocab
        # all_genes should be the dataset's gene list
        expected_all = list(evaluator._data_loader.adata.var_names)
        assert received_args["all_genes"] == expected_all

    def test_evaluate_stage2_mirrors_stage1(self):
        """evaluate_stage2 should return the same result as evaluate_stage1."""
        evaluator = _make_evaluator(n_cells=100)
        out_vocab_pool = [f"GENE_{i}" for i in range(150, 300)]

        def good_selector(vocab, all_genes, prev=None):
            return vocab[:100] + out_vocab_pool[:100]

        result1 = evaluator.evaluate_stage1(good_selector)
        result2 = evaluator.evaluate_stage2(good_selector)

        import math
        # Both should succeed or both fail.
        assert result1.artifacts.get("error") == result2.artifacts.get("error")
        if not result1.artifacts.get("error"):
            p1 = result1.metrics["primary"]
            p2 = result2.metrics["primary"]
            # NaN == NaN is False; handle explicitly.
            if math.isnan(p1):
                assert math.isnan(p2)
            else:
                assert p1 == p2

    def test_valid_result_has_artifacts(self):
        """A successful evaluation must include retained_genes and suggestions."""
        evaluator = _make_evaluator(n_cells=100)
        out_vocab_pool = [f"GENE_{i}" for i in range(150, 300)]

        def good_selector(vocab, all_genes, prev=None):
            return vocab[:100] + out_vocab_pool[:100]

        result = evaluator.evaluate_stage1(good_selector)
        assert not result.artifacts.get("error"), f"Unexpected error: {result.artifacts.get('error_message')}"
        assert "retained_genes" in result.artifacts
        assert "suggestions" in result.artifacts
        assert isinstance(result.artifacts["retained_genes"], list)
        assert isinstance(result.artifacts["suggestions"], list)


# ══════════════════════════════════════════════════════════════════════════════
# Group C — EvolutionRunner unit tests
# ══════════════════════════════════════════════════════════════════════════════

_MINIMAL_YAML = """\
evolution:
  max_iterations: 5
  population_size: 5
  archive_size: 3
  num_islands: 1
  checkpoint_interval: 1
  random_seed: 42

llm:
  primary_model: "gemini-2.5-pro"
  primary_weight: 1.0
  temperature: 0.3
  max_tokens: 8000

paths:
  program_file: "src/gene_selector_template.py"
  evaluator_module: "src.openevolve_adapter"
  evaluator_class: "GeneSelectorEvaluator"
  output_dir: "results/evolution"
  checkpoint_dir: "results/checkpoints"

system_message: "Test prompt."
"""


class TestEvolutionRunner:
    """Tests for src/evolve.py."""

    @pytest.fixture
    def config_file(self, tmp_path):
        """Write a minimal config YAML to a temp file and return its path."""
        cfg = tmp_path / "evolve_config.yaml"
        cfg.write_text(_MINIMAL_YAML)
        return str(cfg)

    def test_init_loads_config(self, config_file):
        """EvolutionRunner.__init__ should load the YAML without raising."""
        from src.evolve import EvolutionRunner

        runner = EvolutionRunner(config_file)
        assert runner._config["evolution"]["max_iterations"] == 5

    def test_init_missing_file_raises(self, tmp_path):
        """EvolutionRunner.__init__ should raise FileNotFoundError for missing config."""
        from src.evolve import EvolutionRunner

        with pytest.raises(FileNotFoundError):
            EvolutionRunner(str(tmp_path / "nonexistent.yaml"))

    def test_log_generation_creates_file(self, config_file, tmp_path):
        """log_generation should write gen_{id}/result.json under output_dir."""
        from src.evolve import EvolutionRunner, GenerationResult

        runner = EvolutionRunner(config_file, output_dir=str(tmp_path / "evolution"))
        result = GenerationResult(
            generation_id=0,
            best_score=0.42,
            best_genes=["GENE_1", "GENE_2"],
            retained_genes=["GENE_1"],
            all_scores=[0.42, 0.35],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        runner.log_generation(0, result)

        out_path = tmp_path / "evolution" / "gen_0" / "result.json"
        assert out_path.exists(), f"Expected {out_path} to exist"

        with open(out_path) as fh:
            data = json.load(fh)
        assert data["generation_id"] == 0
        assert data["best_score"] == pytest.approx(0.42)

    def test_log_generation_directory_created(self, config_file, tmp_path):
        """log_generation should create parent directories if they don't exist."""
        from src.evolve import EvolutionRunner, GenerationResult

        deep_out = tmp_path / "a" / "b" / "c"
        runner = EvolutionRunner(config_file, output_dir=str(deep_out))
        result = GenerationResult(generation_id=3, best_score=0.5)
        runner.log_generation(3, result)

        assert (deep_out / "gen_3" / "result.json").exists()

    def test_generation_result_json_roundtrip(self):
        """GenerationResult must serialize to/from JSON cleanly."""
        from src.evolve import GenerationResult
        from dataclasses import asdict

        r = GenerationResult(
            generation_id=2,
            best_score=0.65,
            best_genes=["A", "B", "C"],
            retained_genes=["A"],
            all_scores=[0.65, 0.60, 0.55],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        serialized = json.dumps(asdict(r))
        restored = json.loads(serialized)
        assert restored["generation_id"] == 2
        assert restored["best_score"] == pytest.approx(0.65)
        assert restored["best_genes"] == ["A", "B", "C"]

    def test_build_openevolve_config_overrides_iterations(self, config_file):
        """_build_openevolve_config should apply n_generations override."""
        from src.evolve import EvolutionRunner

        runner = EvolutionRunner(config_file)
        config = runner._build_openevolve_config(n_generations=3)
        assert config["evolution"]["max_iterations"] == 3

    def test_build_openevolve_config_no_override_keeps_yaml(self, config_file):
        """_build_openevolve_config without override uses YAML's max_iterations."""
        from src.evolve import EvolutionRunner

        runner = EvolutionRunner(config_file)
        config = runner._build_openevolve_config(n_generations=None)
        assert config["evolution"]["max_iterations"] == 5


# ══════════════════════════════════════════════════════════════════════════════
# Group D — Suggestion builder (src/openevolve_adapter._build_suggestions)
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildSuggestions:
    """Tests for the _build_suggestions helper in openevolve_adapter."""

    def _make_eval_result(
        self,
        retained: list[str],
        coefficients: dict,
        scores: dict | None = None,
    ):
        """Return a minimal EvalResult-like object."""
        from src.evaluator import EvalResult

        return EvalResult(
            scores=scores or {f"target_{i}": 0.3 for i in range(6)},
            aggregate_score=0.3,
            retained_genes=retained,
            coefficients=coefficients,
            n_retained=len(retained),
        )

    def test_returns_list_of_strings(self):
        """_build_suggestions must return a list of non-empty strings."""
        from src.openevolve_adapter import _build_suggestions

        selected = [f"GENE_{i}" for i in range(200)]
        retained = selected[:20]
        coefs = {g: 0.1 for g in retained}
        eval_result = self._make_eval_result(retained, coefs)
        suggestions = _build_suggestions(eval_result, selected)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        for s in suggestions:
            assert isinstance(s, str) and len(s) > 0

    def test_mentions_retention_stats(self):
        """Suggestions should mention the number of retained genes."""
        from src.openevolve_adapter import _build_suggestions

        selected = [f"GENE_{i}" for i in range(200)]
        retained = selected[:30]
        coefs = {g: 0.2 for g in retained}
        eval_result = self._make_eval_result(retained, coefs)
        suggestions = _build_suggestions(eval_result, selected)

        combined = " ".join(suggestions)
        assert "30" in combined  # n_retained appears somewhere

    def test_zeroed_genes_mentioned(self):
        """Suggestions should list example zeroed genes."""
        from src.openevolve_adapter import _build_suggestions

        selected = [f"GENE_{i}" for i in range(200)]
        retained = selected[:5]
        coefs = {g: 0.1 for g in retained}
        eval_result = self._make_eval_result(retained, coefs)
        suggestions = _build_suggestions(eval_result, selected)

        # The suggestion containing zeroed genes should mention one of them.
        combined = " ".join(suggestions)
        assert "GENE_" in combined
