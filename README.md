# DarwinSeq

Agentic gene selection for the SEA-AD DREAM Challenge using evolutionary search.

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Run the Baseline

Establishes random gene-selection performance across 50 seeds (no LLM guidance).

```bash
python3 scripts/run_baseline.py
```

Default settings: 50 seeds · 100 in-vocab + 100 out-of-vocab genes · 100 cells per seed.

Override any parameter:

```bash
python3 scripts/run_baseline.py \
  --n_seeds 50 \
  --n_in 100 \
  --n_out 100 \
  --n_cells 100 \
  --output_dir results/baseline/ \
  --data_path /path/to/SEAAD_A9_RNAseq_DREAM.h5ad
```

Results are saved to `results/baseline/`:
- `all_runs.json` — per-seed scores, gene lists, retained genes
- `summary.json` — mean ± std aggregate score, gene retention frequency
- `best_run.json` — highest-scoring random run

## Run Tests

```bash
python3 -m pytest tests/ -v
```

## Project Structure

```
config/          LASSO hyperparameters, gene vocabulary
scripts/         run_baseline.py and utility scripts
src/             data_loader, gene_vocab, sampler, evaluator
tests/           unit tests (no real data required)
results/         output from runs (gitignored)
```
