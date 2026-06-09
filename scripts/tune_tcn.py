"""Greedy TCN tuning on fold-selectable 5-fold GroupKFold.

Default mode runs fold 0 of the project 5-fold split for fast manual tuning.
Before promoting parameters, rerun with ``--folds all`` and transfer the printed
``scripts/run_train.py`` command to the final Kaggle train flow.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.metrics import rmse
from rogii.train import _collect_train_sequences, _make_dataloader
from rogii.tcn_model import TCNModel
from rogii.validation import validate_no_group_overlap


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = str(REPO_ROOT / "data")
DEFAULT_CHANNEL_CANDIDATES = "16,32;"
DEFAULT_LR_CANDIDATES = "1e-4,3e-4,5e-4,7e-4"


@dataclass(frozen=True)
class FoldResult:
    fold_idx: int
    best_epoch: int
    best_rmse_delta: float
    best_val_mse: float


@dataclass(frozen=True)
class ConfigResult:
    channels: list[int]
    lr: float
    fold_results: list[FoldResult]

    @property
    def mean_rmse(self) -> float:
        return float(np.mean([r.best_rmse_delta for r in self.fold_results]))

    @property
    def std_rmse(self) -> float:
        return float(np.std([r.best_rmse_delta for r in self.fold_results]))

    @property
    def fold_scores(self) -> list[float]:
        return [r.best_rmse_delta for r in self.fold_results]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA, help="Competition data directory")
    parser.add_argument("--folds", default="0", help="Fold ids to run, e.g. '0', '0,1', or 'all'")
    parser.add_argument("--n-splits", type=int, default=5, help="Project GroupKFold split count")
    parser.add_argument("--window", type=int, default=16, help="TCN sliding window size")
    parser.add_argument("--stride", type=int, default=4, help="Train sliding-window stride")
    parser.add_argument("--val-stride", type=int, default=1, help="Validation stride for RMSE monitoring")
    parser.add_argument("--batch-size", type=int, default=2048, help="DataLoader batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per config/fold")
    parser.add_argument("--patience", type=int, default=2, help="Early-stop patience on val RMSE; 0 disables")
    parser.add_argument("--weight-decay", type=float, default=2e-4, help="AdamW weight decay")
    parser.add_argument("--kernel-size", type=int, default=3, help="TCN kernel size")
    parser.add_argument("--dropout", type=float, default=0.1, help="TCN dropout")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Training device")
    parser.add_argument("--channels", help="Run one channel config, e.g. '64,128,256'")
    parser.add_argument("--lr", type=float, help="Run one learning rate with --channels")
    parser.add_argument("--channel-candidates", default=DEFAULT_CHANNEL_CANDIDATES,
                        help="Semicolon-separated channel candidates")
    parser.add_argument("--channel-lr", type=float, default=3e-4,
                        help="Learning rate used while comparing channel candidates")
    parser.add_argument("--lr-candidates", default=DEFAULT_LR_CANDIDATES,
                        help="Comma-separated LR candidates for the best channels")
    args = parser.parse_args()

    if (args.channels is None) != (args.lr is None):
        parser.error("--channels and --lr must be provided together for single-config mode")
    if args.n_splits < 2:
        parser.error("--n-splits must be at least 2")
    if args.window < 1 or args.stride < 1 or args.val_stride < 1:
        parser.error("--window, --stride, and --val-stride must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    return args


def parse_channels(value: str) -> list[int]:
    channels = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not channels:
        raise ValueError("Channel list must not be empty")
    return channels


def parse_channel_candidates(value: str) -> list[list[int]]:
    return [parse_channels(chunk) for chunk in value.split(";") if chunk.strip()]


def parse_lr_candidates(value: str) -> list[float]:
    lrs = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not lrs:
        raise ValueError("LR candidate list must not be empty")
    return lrs


def parse_fold_ids(value: str, n_splits: int) -> list[int]:
    if value.lower() == "all":
        return list(range(n_splits))
    fold_ids = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not fold_ids:
        raise ValueError("At least one fold id is required")
    bad = [f for f in fold_ids if f < 0 or f >= n_splits]
    if bad:
        raise ValueError(f"Fold ids out of range for n_splits={n_splits}: {bad}")
    return fold_ids


def load_and_norm(data_dir: str):
    """Load once and apply prediction-time-compatible per-well X normalization."""
    print("Loading data...", flush=True)
    sequences, _feature_columns = _collect_train_sequences(data_dir, residual_target=True)
    print(f"  {len(sequences)} wells, per-well X norm...", flush=True)

    for s in sequences:
        x = s.X.astype(np.float64)
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.clip(std, 1e-8, None)
        s.X = torch.from_numpy(((x - mean) / std).astype(np.float32)).T.contiguous()

    return sequences


def build_group_folds(sequences, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.arange(len(sequences), dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in splitter.split(np.zeros(len(sequences)), groups=groups):
        validate_no_group_overlap(set(groups[train_idx]), set(groups[val_idx]))
        folds.append((train_idx, val_idx))
    return folds


def fit_target_scaler(sequences, indices: np.ndarray) -> StandardScaler | None:
    y = np.concatenate([np.asarray(sequences[i].y, dtype=np.float64) for i in indices])
    if np.std(y) < 1e-12:
        return None
    scaler = StandardScaler()
    scaler.fit(y.reshape(-1, 1))
    return scaler


def scaled_sequences(sequences, indices: np.ndarray, scaler: StandardScaler | None):
    out = []
    for i in indices:
        y = np.asarray(sequences[i].y, dtype=np.float64)
        if scaler is not None:
            y = scaler.transform(y.reshape(-1, 1)).ravel()
        out.append(replace(sequences[i], y=torch.from_numpy(y.astype(np.float32))))
    return out


def inverse_target(values: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    if scaler is None:
        return values
    return scaler.inverse_transform(values.reshape(-1, 1)).ravel()


def evaluate_validation(model, val_loader, criterion, target_scaler, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    pred_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            total_loss += criterion(pred, yb).item() * xb.size(0)
            pred_parts.append(pred.detach().cpu().numpy().ravel())
            true_parts.append(yb.detach().cpu().numpy().ravel())

    val_mse = total_loss / len(val_loader.dataset)
    y_pred_scaled = np.concatenate(pred_parts)
    y_true_scaled = np.concatenate(true_parts)
    val_rmse_delta = rmse(inverse_target(y_true_scaled, target_scaler), inverse_target(y_pred_scaled, target_scaler))
    return val_mse, val_rmse_delta


def run_one_fold(sequences, train_idx: np.ndarray, val_idx: np.ndarray, fold_idx: int,
                 channels: list[int], lr: float, args: argparse.Namespace,
                 device: str) -> FoldResult:
    seed = args.seed + fold_idx
    torch.manual_seed(seed)
    np.random.seed(seed)

    target_scaler = fit_target_scaler(sequences, train_idx)
    train_seqs = scaled_sequences(sequences, train_idx, target_scaler)
    val_seqs = scaled_sequences(sequences, val_idx, target_scaler)

    train_loader = _make_dataloader(train_seqs, args.window, args.batch_size, shuffle=True, stride=args.stride)
    val_loader = _make_dataloader(val_seqs, args.window, args.batch_size, shuffle=False, stride=args.val_stride)
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError(
            f"Fold {fold_idx} produced empty train/val windows. "
            f"Try a smaller --window or inspect sequence lengths."
        )

    input_size = sequences[0].X.shape[0]
    model = TCNModel(
        input_size=int(input_size),
        num_channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6,
    )
    criterion = torch.nn.MSELoss()
    use_amp = device == "cuda"
    amp = torch.amp.GradScaler("cuda") if use_amp else None

    best_rmse = float("inf")
    best_mse = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    print(
        f"\nfold={fold_idx + 1}/{args.n_splits}, ch={channels}, lr={lr:.0e}, "
        f"train={len(train_idx)} wells, val={len(val_idx)} wells",
        flush=True,
    )
    print("  ep  train_mse  val_mse  best_mse  rmse_delta  best_rmse      lr   time", flush=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(xb), yb)
                amp.scale(loss).backward()
                amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp.step(optimizer)
                amp.update()
            else:
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_mse, val_rmse = evaluate_validation(model, val_loader, criterion, target_scaler, device)
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]["lr"]

        best_mse = min(best_mse, val_mse)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch + 1
            epochs_no_improve = 0
            marker = "*"
        else:
            epochs_no_improve += 1
            marker = " "

        print(
            f"  {epoch + 1:>2}{marker}  {train_loss:>9.4f} {val_mse:>8.4f} {best_mse:>9.4f} "
            f"{val_rmse:>10.4f} {best_rmse:>10.4f} {current_lr:>8.2e} {time.time() - t0:>5.0f}s",
            flush=True,
        )

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"  Early stop: best epoch {best_epoch}, best_rmse={best_rmse:.4f}", flush=True)
            break

    return FoldResult(fold_idx=fold_idx, best_epoch=best_epoch, best_rmse_delta=best_rmse, best_val_mse=best_mse)


def evaluate_config(sequences, folds, fold_ids: list[int], channels: list[int], lr: float,
                    args: argparse.Namespace, device: str) -> ConfigResult:
    fold_results = []
    for fold_idx in fold_ids:
        train_idx, val_idx = folds[fold_idx]
        fold_results.append(run_one_fold(sequences, train_idx, val_idx, fold_idx, channels, lr, args, device))

    result = ConfigResult(channels=list(channels), lr=lr, fold_results=fold_results)
    fold_scores = [round(s, 4) for s in result.fold_scores]
    print(
        f"  >> ch={channels}, lr={lr:.0e}, mean_rmse={result.mean_rmse:.4f}, "
        f"std={result.std_rmse:.4f}, folds={fold_scores}",
        flush=True,
    )
    return result


def print_run_train_command(channels: list[int], lr: float, args: argparse.Namespace) -> None:
    ch = ",".join(str(c) for c in channels)
    cmd = (
        f"python scripts/run_train.py --model-type tcn --config configs/a5_tcn.yaml "
        f"--data-dir data --save-oof --tcn-channels {ch} --tcn-lr {lr} "
        f"--tcn-window {args.window} --tcn-stride {args.stride} "
        f"--tcn-batch-size {args.batch_size} --tcn-epochs <FINAL_EPOCHS> "
        f"--tcn-kernel-size {args.kernel_size} --tcn-dropout {args.dropout} "
        f"--tcn-patience {args.patience if args.patience > 0 else 5}"
    )
    print(f"\n=== RECOMMENDATION ===\n  --tcn-channels {ch}\n  --tcn-lr {lr}\n  final train command:\n  {cmd}")
    if args.folds.lower() != "all":
        print("  before promotion: rerun this tuner with --folds all")


def main() -> None:
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    print("=== TCN Greedy Search ===")
    print(
        f"  device={device}, folds={args.folds}/{args.n_splits}, w={args.window}, "
        f"train_stride={args.stride}, val_stride={args.val_stride}, B={args.batch_size}, ep={args.epochs}",
        flush=True,
    )

    sequences = load_and_norm(args.data_dir)
    folds = build_group_folds(sequences, args.n_splits)
    fold_ids = parse_fold_ids(args.folds, args.n_splits)
    print(f"  selected folds: {[f + 1 for f in fold_ids]}", flush=True)

    if args.channels is not None:
        channels = parse_channels(args.channels)
        result = evaluate_config(sequences, folds, fold_ids, channels, args.lr, args, device)
        print_run_train_command(result.channels, result.lr, args)
        return

    print("\n-- Step 1: fixed lr, vary channels --")
    channel_results: list[ConfigResult] = []
    for channels in parse_channel_candidates(args.channel_candidates):
        channel_results.append(evaluate_config(sequences, folds, fold_ids, channels, args.channel_lr, args, device))

    best_channel_result = min(channel_results, key=lambda r: r.mean_rmse)
    best_channels = best_channel_result.channels
    print(
        f"\nBest channels: {best_channels} "
        f"(mean_rmse={best_channel_result.mean_rmse:.4f}, std={best_channel_result.std_rmse:.4f})",
        flush=True,
    )

    print(f"\n-- Step 2: ch={best_channels}, vary lr --")
    lr_results: list[ConfigResult] = []
    for lr in parse_lr_candidates(args.lr_candidates):
        lr_results.append(evaluate_config(sequences, folds, fold_ids, best_channels, lr, args, device))

    best_lr_result = min(lr_results, key=lambda r: r.mean_rmse)
    print(
        f"\nBest lr: {best_lr_result.lr:.0e} "
        f"(mean_rmse={best_lr_result.mean_rmse:.4f}, std={best_lr_result.std_rmse:.4f})",
        flush=True,
    )
    print_run_train_command(best_lr_result.channels, best_lr_result.lr, args)


if __name__ == "__main__":
    main()
