"""Compatibility entry point for offline Kaggle inference."""

from pathlib import Path
import runpy

def main() -> None:
    runpy.run_path(str(Path(__file__).with_name("kaggle_offline_inference.py")), run_name="__main__")


if __name__ == "__main__":
    main()
