"""
verify_data.py — Verify the structure and integrity of the SEA-AD .h5ad file.

Loads metadata (obs/var) via safe_load(), then prints a structured summary
of shape, metadata columns, gene space, expression type, and cell-type
hierarchy. Expression values are sampled via h5py (no full matrix load).

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --data_path /path/to/other.h5ad
"""

import argparse
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import DEFAULT_DATA_PATH, safe_load

# Expected metadata columns in adata.obs.
_EXPECTED_OBS_COLS = [
    "Age at Death",
    "Sex",
    "APOE Genotype",
    "Thal",
    "Braak",
    "CERAD",
    "ADNC",
    "percent 6e10 positive area",
    "percent AT8 positive area",
    "percent NeuN positive area",
    "percent GFAP positive area",
    "percent pTDP43 positive area",
    "percent aSyn positive area",
]

# Cell-type hierarchy columns.
_CELLTYPE_COLS = ["Class", "Subclass", "Supertype"]


def _section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def verify(data_path: str) -> None:
    """
    Load metadata via safe_load() and print a structured verification summary.

    Samples a 5-cell slice of X directly via h5py to confirm float values
    without loading the full matrix into memory.

    Args:
        data_path: Path to the .h5ad file to verify.
    """
    adata = safe_load(h5ad_path=data_path)

    _section("Shape")
    print(f"  Cells × Genes: {adata.n_obs:,} × {adata.n_vars:,}")

    _section("Metadata columns (adata.obs)")
    present = list(adata.obs.columns)
    print(f"  Total columns : {len(present)}")
    print(f"  All columns   : {present}")
    missing = [c for c in _EXPECTED_OBS_COLS if c not in present]
    if missing:
        print(f"\n  MISSING expected columns: {missing}")
    else:
        print(f"\n  All {len(_EXPECTED_OBS_COLS)} expected columns are present.")

    _section("Gene space (adata.var)")
    print(adata.var.head().to_string())
    print(f"\n  var_names sample: {list(adata.var_names[:5])}")
    print(f"  Total genes     : {adata.n_vars:,}")

    _section("Expression matrix (adata.X) — sampled 5 cells via h5py")
    with h5py.File(data_path, "r") as f:
        indptr = f["X/indptr"][:]
        dtype = f["X/data"].dtype
        print(f"  dtype     : {dtype}")
        print(f"  n_cells   : {len(indptr) - 1:,}")
        print(f"  n_nonzeros: {indptr[-1]:,}")

        # Read a few values from cell 0 as a spot check.
        ds, de = int(indptr[0]), int(indptr[1])
        sample_vals = f["X/data"][ds:de][:10]
        print(f"  Sample values (cell 0): {sample_vals}")

        all_int = bool(np.all(sample_vals == np.floor(sample_vals)))
        if all_int:
            print("\n  WARNING: values look like integers — expected log-normalized floats.")
        else:
            print("\n  OK: values appear to be log-normalized floats.")

    _section("Layers")
    with h5py.File(data_path, "r") as f:
        if "layers" in f:
            layer_names = list(f["layers"].keys())
            print(f"  Available layers: {layer_names}")
            if "UMIs" in f["layers"]:
                umi_dtype = f["layers/UMIs/data"].dtype
                umi_nnz = f["layers/UMIs/indptr"][-1]
                print(f"  UMIs dtype: {umi_dtype}, n_nonzeros: {int(umi_nnz):,}")
            else:
                print("  WARNING: 'UMIs' layer not found.")
        else:
            print("  WARNING: no layers group found.")

    _section("Cell-type hierarchy")
    for col in _CELLTYPE_COLS:
        if col in adata.obs.columns:
            uniq = sorted(adata.obs[col].dropna().unique())
            preview = uniq[:10]
            suffix = "..." if len(uniq) > 10 else ""
            print(f"  {col} ({len(uniq)} unique): {preview}{suffix}")
        else:
            print(f"  WARNING: column '{col}' not found in adata.obs.")

    _section("Corruption report")
    import json, os
    report_path = "data/corruption_report.json"
    if os.path.exists(report_path):
        with open(report_path) as fh:
            report = json.load(fh)
        print(f"  Original cells : {report['original_n_samples']:,}")
        print(f"  Dropped        : {report['dropped_n_samples']:,}")
        print(f"  Clean cells    : {report['clean_n_samples']:,}")
        print(f"  Drop rate      : {report['drop_rate_percent']:.4f}%")

    _section("Done")
    print(f"  Verification complete for: {data_path}\n")


def main() -> None:
    """Parse CLI arguments and run verification."""
    parser = argparse.ArgumentParser(
        description="Verify the structure and integrity of a SEA-AD .h5ad file."
    )
    parser.add_argument(
        "--data_path",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the .h5ad file. Defaults to: {DEFAULT_DATA_PATH}",
    )
    args = parser.parse_args()
    verify(args.data_path)


if __name__ == "__main__":
    main()
