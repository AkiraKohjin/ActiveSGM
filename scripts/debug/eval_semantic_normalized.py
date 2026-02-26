#!/usr/bin/env python3
"""
Evaluate semantic rendering with normalized top-k logits (debug only).

Purpose
- Test whether lack of probability normalization is causing noisy semantic renderings
  and low mIoU, without changing any training code.

Behavior
- Loads the existing checkpoint params*.npz via the same code path as evaluation.
- Replaces semantic logits (top-k values per Gaussian) with a *positive-normalized*
  version so each Gaussian's top-k distribution sums to 1.
- Runs the standard eval_semantic pipeline and writes results to a separate
  eval directory suffix so existing results are not overwritten.

This script is intentionally verbose and heavily commented for safe debugging.
"""

import argparse
import os
import sys

import torch
from tensorboardX import SummaryWriter

# Ensure local imports work the same way as the main entry points.
sys.path.append(os.getcwd())

from src.naruto.cfg_loader import load_cfg
from src.utils.general_utils import fix_random_seed, InfoPrinter
from src.slam import init_SLAM_model


def _positive_normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize each row to sum to 1 after clipping negative values.

    This mirrors the paper's "clip to non-negative and normalize" semantics
    while keeping the top-k *class IDs* unchanged (only scaling the values).
    """
    x = torch.clamp(x, min=0)
    denom = x.sum(dim=1, keepdim=True)
    denom = torch.clamp(denom, min=eps)
    return x / denom


def _log_stats(tag: str, t: torch.Tensor) -> None:
    """Print simple tensor stats to help compare before/after normalization."""
    with torch.no_grad():
        finite = torch.isfinite(t)
        bad = int((~finite).sum().item())
        if finite.any():
            vals = t[finite]
            vmin = float(vals.min().item())
            vmax = float(vals.max().item())
            vmean = float(vals.mean().item())
        else:
            vmin = vmax = vmean = float("nan")
        print(
            "[eval_semantic_normalized] "
            f"{tag}: bad={bad} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic rendering with normalized top-k logits"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/default.py",
        help="NARUTO config",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Override result directory (optional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed; also used as initial pose idx for Replica",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="final",
        help="Checkpoint stage (SplaTAM) to load",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1100,
        help="Checkpoint step to load (SplaTAM) for params*.npz",
    )
    parser.add_argument(
        "--out_suffix",
        type=str,
        default="final_norm",
        help="Suffix for eval output dir (e.g., eval_final_norm)",
    )
    parser.add_argument(
        "--save_frames",
        type=int,
        default=1,
        help="Whether to save per-frame eval visualizations",
    )
    return parser.parse_args()


def main() -> None:
    info_printer = InfoPrinter("ActiveSem")
    args = _parse_args()

    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)

    # Align seed handling with the standard eval script.
    fix_random_seed(main_cfg.general.seed)

    # Init logger (same as eval_semantic.py)
    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    _ = SummaryWriter(f"{log_savedir}")

    # Initialize model and load checkpoint.
    slam = init_SLAM_model(main_cfg, info_printer, None)
    slam.load_params_by_step(step=args.step, stage=args.stage)

    # Sanity-check semantic logits shape: [N, TOPK].
    sem = slam.params.get("semantic_logits", None)
    if sem is None:
        raise RuntimeError("semantic_logits not found in loaded params")

    _log_stats("semantic_logits (raw)", sem)

    # Normalize per-Gaussian top-k values to form a probability distribution.
    # This does NOT change the top-k class IDs (seman_cls_ids).
    sem_norm = _positive_normalize_rows(sem)
    _log_stats("semantic_logits (normalized)", sem_norm)

    # Replace in params for evaluation (no training updates).
    slam.params["semantic_logits"] = torch.nn.Parameter(sem_norm)

    # Run evaluation; results go to eval_<out_suffix> directory.
    slam.eval_semantic_result(
        eval_dir_suffix=args.out_suffix,
        ignore_first_frame=True,
        save_frames=bool(args.save_frames),
    )


if __name__ == "__main__":
    main()
