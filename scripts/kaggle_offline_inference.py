"""Run offline Kaggle inference with robust mounted-input discovery."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.kaggle_runtime import (
    DEFAULT_COMPETITION_SLUG,
    DEFAULT_MODEL_DATASET_SLUG,
    DEFAULT_MODEL_FILE,
    DEFAULT_REPO_DATASET_SLUG,
    build_predict_command,
    require_nonempty_file,
    resolve_inference_paths,
)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input", help="Kaggle input root")
    parser.add_argument("--output", default="/kaggle/working/submission.csv", help="Submission output path")
    parser.add_argument("--repo-dataset-slug", default=DEFAULT_REPO_DATASET_SLUG)
    parser.add_argument("--model-dataset-slug", default=DEFAULT_MODEL_DATASET_SLUG)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--competition-slug", default=DEFAULT_COMPETITION_SLUG)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    paths = resolve_inference_paths(
        input_root=args.input_root,
        output_path=args.output,
        repo_dataset_slug=args.repo_dataset_slug,
        model_dataset_slug=args.model_dataset_slug,
        model_file=args.model_file,
        competition_slug=args.competition_slug,
    )

    print(f"Repo root: {paths.repo_root}")
    print(f"Data dir: {paths.data_dir}")
    print(f"Model path: {paths.model_path}")
    print(f"Output path: {paths.output_path}")

    command = build_predict_command(paths)
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=paths.repo_root, check=True)

    output = require_nonempty_file(paths.output_path)
    print(f"Submission ready: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
