"""Inspect TVT value range across all train horizontal wells for clipping bounds."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rogii.data_loading import list_well_ids, read_horizontal_well


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Competition data directory")
    return parser


def main() -> None:
    args = parse_args().parse_args()

    well_ids = list_well_ids(args.data_dir, "train")
    print(f"Scanning {len(well_ids)} train wells for TVT values ...")

    all_tvt: list[float] = []

    for i, wid in enumerate(well_ids):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(well_ids)} wells scanned, {len(all_tvt)} TVT values collected")
        df = read_horizontal_well(args.data_dir, "train", wid)
        if "TVT" not in df.columns:
            continue
        tvt = df["TVT"].dropna().to_numpy(dtype=float)
        all_tvt.extend(tvt.tolist())

    if not all_tvt:
        print("No TVT values found")
        return

    arr = np.array(all_tvt, dtype=float)
    percentiles = [0.01, 0.1, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9, 99.99]
    values = np.percentile(arr, percentiles)

    print(f"\nTotal TVT values: {len(arr):,}")
    print(f"  Min:    {arr.min():.2f}")
    print(f"  Max:    {arr.max():.2f}")
    print(f"  Mean:   {arr.mean():.2f}")
    print(f"  Std:    {arr.std():.2f}")
    print()

    for p, v in zip(percentiles, values):
        print(f"  p{p:6.3f}: {v:10.2f}")

    print()
    print("Suggested clipping bounds:")
    for low_p, high_p in [(0.1, 99.9), (0.5, 99.5), (1.0, 99.0)]:
        low = float(np.percentile(arr, low_p))
        high = float(np.percentile(arr, high_p))
        outside = float(np.mean((arr < low) | (arr > high))) * 100
        print(f"  p{low_p}-p{high_p}:  [{low:.2f}, {high:.2f}]  ({outside:.3f}% outside)")


if __name__ == "__main__":
    main()
