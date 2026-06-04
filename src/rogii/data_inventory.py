"""Data inventory helpers for local and Kaggle data directories."""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from rogii.data_loading import HORIZONTAL_SUFFIX, TYPEWELL_SUFFIX, parse_well_id, require_data_dir


def list_data_files(data_dir: str | Path) -> list[Path]:
    """List files under a data directory without reading their contents."""
    path = require_data_dir(data_dir)
    return sorted(item for item in path.rglob("*") if item.is_file())


def _csv_header_and_rows(path: Path) -> tuple[tuple[str, ...], int]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader))
        rows = sum(1 for _ in reader)
    return header, rows


def _row_stats(values: list[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "median": int(median(values)),
        "max": max(values),
    }


def build_inventory(data_dir: str | Path) -> dict[str, object]:
    """Build a compact inventory from the competition data directory."""
    root = require_data_dir(data_dir)
    files = list_data_files(root)
    csv_files = sorted(path for path in files if path.suffix.lower() == ".csv")
    inventory: dict[str, object] = {
        "data_dir": str(root),
        "file_count": len(files),
        "csv_count": len(csv_files),
        "extensions": dict(Counter(path.suffix.lower() or "<none>" for path in files)),
        "splits": {},
    }

    sample_path = root / "sample_submission.csv"
    if sample_path.is_file():
        header, rows = _csv_header_and_rows(sample_path)
        inventory["sample_submission"] = {"columns": list(header), "rows": rows}

    for split in ("train", "test"):
        split_path = root / split
        if not split_path.is_dir():
            continue
        split_csvs = sorted(split_path.glob("*.csv"))
        well_kinds: dict[str, set[str]] = defaultdict(set)
        header_counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
        rows_by_kind: dict[str, list[int]] = defaultdict(list)
        for path in split_csvs:
            well_id = parse_well_id(path)
            if path.name.endswith(HORIZONTAL_SUFFIX):
                kind = "horizontal_well"
            elif path.name.endswith(TYPEWELL_SUFFIX):
                kind = "typewell"
            else:
                kind = "unknown"
            well_kinds[well_id].add(kind)
            header, rows = _csv_header_and_rows(path)
            header_counts[(kind, header)] += 1
            rows_by_kind[kind].append(rows)

        inventory["splits"][split] = {
            "csv_files": len(split_csvs),
            "unique_wells": len(well_kinds),
            "well_kind_counts": {
                "+".join(kinds): count
                for kinds, count in Counter(tuple(sorted(value)) for value in well_kinds.values()).items()
            },
            "headers": [
                {"kind": kind, "count": count, "columns": list(header)}
                for (kind, header), count in header_counts.most_common()
            ],
            "row_stats": {kind: _row_stats(values) for kind, values in rows_by_kind.items()},
        }

    return inventory


def format_inventory(inventory: dict[str, object]) -> str:
    """Format inventory as a concise text report."""
    lines = [
        f"Data directory: {inventory['data_dir']}",
        f"Files: {inventory['file_count']}",
        f"CSV files: {inventory['csv_count']}",
        f"Extensions: {inventory['extensions']}",
    ]
    sample = inventory.get("sample_submission")
    if isinstance(sample, dict):
        lines.append(f"Sample submission: rows={sample['rows']} columns={sample['columns']}")
    splits = inventory.get("splits", {})
    if isinstance(splits, dict):
        for split, details in splits.items():
            if not isinstance(details, dict):
                continue
            lines.append("")
            lines.append(f"[{split}]")
            lines.append(f"CSV files: {details['csv_files']}")
            lines.append(f"Unique wells: {details['unique_wells']}")
            lines.append(f"Well kind counts: {details['well_kind_counts']}")
            for header in details["headers"]:
                lines.append(f"Header {header['kind']} x{header['count']}: {header['columns']}")
            for kind, stats in details["row_stats"].items():
                lines.append(f"Rows {kind}: {stats}")
    return "\n".join(lines)
