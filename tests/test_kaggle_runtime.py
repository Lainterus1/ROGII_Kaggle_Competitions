from pathlib import Path

import pytest

from rogii.kaggle_runtime import (
    KaggleInferencePaths,
    build_predict_command,
    find_competition_data_dir,
    find_model_path,
    find_repo_root,
    require_nonempty_file,
    resolve_inference_paths,
)


def _make_repo(root: Path) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "src" / "rogii").mkdir(parents=True)
    (root / "scripts" / "run_predict.py").write_text("print('predict')\n", encoding="utf-8")


def test_find_repo_root_accepts_flat_dataset_layout(tmp_path: Path) -> None:
    repo = tmp_path / "rogii-repo-v2"
    _make_repo(repo)

    assert find_repo_root(tmp_path) == repo


def test_find_repo_root_accepts_nested_notebook_output_layout(tmp_path: Path) -> None:
    repo = tmp_path / "datasets" / "daniilgonchar" / "rogii-repo" / "ROGII_Kaggle_Competitions"
    _make_repo(repo)

    assert find_repo_root(tmp_path, preferred_slug="rogii-repo") == repo


def test_find_competition_data_dir_requires_only_test_and_sample(tmp_path: Path) -> None:
    data_dir = tmp_path / "competitions" / "rogii-wellbore-geology-prediction"
    (data_dir / "test").mkdir(parents=True)
    (data_dir / "sample_submission.csv").write_text("id,tvt\na_1,0.0\n", encoding="utf-8")

    assert find_competition_data_dir(tmp_path) == data_dir


def test_find_model_path_prefers_configured_model_dataset_slug(tmp_path: Path) -> None:
    old_model = tmp_path / "rogii-models" / "baseline_lgbm.pkl"
    new_model = tmp_path / "rogii-models-v2" / "baseline_lgbm.pkl"
    old_model.parent.mkdir(parents=True)
    new_model.parent.mkdir(parents=True)
    old_model.write_bytes(b"old")
    new_model.write_bytes(b"new")

    assert find_model_path(tmp_path) == new_model


def test_resolve_inference_paths_combines_all_inputs(tmp_path: Path) -> None:
    repo = tmp_path / "rogii-repo-v2"
    _make_repo(repo)
    data_dir = tmp_path / "rogii-wellbore-geology-prediction"
    (data_dir / "test").mkdir(parents=True)
    (data_dir / "sample_submission.csv").write_text("id,tvt\na_1,0.0\n", encoding="utf-8")
    model = tmp_path / "rogii-models-v2" / "baseline_lgbm.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")

    paths = resolve_inference_paths(tmp_path, tmp_path / "submission.csv")

    assert paths.repo_root == repo
    assert paths.data_dir == data_dir
    assert paths.model_path == model
    assert paths.output_path == tmp_path / "submission.csv"


def test_build_predict_command_uses_resolved_paths(tmp_path: Path) -> None:
    paths = KaggleInferencePaths(
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
        model_path=tmp_path / "model.pkl",
        output_path=tmp_path / "submission.csv",
    )

    command = build_predict_command(paths, python_executable="python")

    assert command == [
        "python",
        str(tmp_path / "repo" / "scripts" / "run_predict.py"),
        "--data-dir",
        str(tmp_path / "data"),
        "--model",
        str(tmp_path / "model.pkl"),
        "--output",
        str(tmp_path / "submission.csv"),
    ]


def test_require_nonempty_file_rejects_missing_and_empty_outputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        require_nonempty_file(tmp_path / "missing.csv")

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        require_nonempty_file(empty)

    valid = tmp_path / "valid.csv"
    valid.write_text("id,tvt\na_1,1.0\n", encoding="utf-8")
    assert require_nonempty_file(valid) == valid
