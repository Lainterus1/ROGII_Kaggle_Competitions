"""Create a compact data inventory for the ROGII competition files."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.data_inventory import build_inventory, format_inventory


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Competition data directory")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    inventory = build_inventory(args.data_dir)
    print(format_inventory(inventory))


if __name__ == "__main__":
    main()
