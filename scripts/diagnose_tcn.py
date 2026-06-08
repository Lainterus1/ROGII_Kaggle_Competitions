"""Phase 0 TCN diagnostic CLI.

Usage:
    python scripts/diagnose_tcn.py --tcn-oof outputs/oof/run_tcn_oof.parquet --data-dir data
    python scripts/diagnose_tcn.py --tcn-oof ... --lgbm-oof ... --data-dir data --n-bins 10
"""

from argparse import ArgumentParser
from pathlib import Path

from rogii.diagnostics import generate_report, load_oof_with_metadata


def main():
    parser = ArgumentParser(description="TCN Phase 0 diagnostic report")
    parser.add_argument("--tcn-oof", required=True, type=str,
                        help="Path to TCN OOF parquet file")
    parser.add_argument("--lgbm-oof", default=None, type=str,
                        help="Path to LGBM OOF parquet file for error correlation analysis")
    parser.add_argument("--data-dir", required=True, type=str,
                        help="Path to data directory (must contain train/ subdirectory)")
    args = parser.parse_args()

    tcn_path = Path(args.tcn_oof)
    if not tcn_path.is_file():
        raise FileNotFoundError(f"TCN OOF not found: {tcn_path}")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    lgbm_path = Path(args.lgbm_oof) if args.lgbm_oof else None
    if lgbm_path is not None and not lgbm_path.is_file():
        raise FileNotFoundError(f"LGBM OOF not found: {lgbm_path}")

    print(f"Loading TCN OOF: {tcn_path}")
    oof = load_oof_with_metadata(tcn_path, data_dir)

    report = generate_report(oof, lgbm_oof_path=lgbm_path)
    print(report)


if __name__ == "__main__":
    main()
