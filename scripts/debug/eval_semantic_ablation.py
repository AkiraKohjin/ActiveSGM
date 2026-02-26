#!/usr/bin/env python3
"""
Semantic evaluation ablations (debug only).

This script runs evaluation without changing training code or checkpoints.
It loads an existing params*.npz checkpoint and modifies *only* the in-memory
parameters for evaluation.

Supported modes
1) onehot:
   - For each Gaussian, convert its top-k semantic logits to a one-hot vector
     at the argmax position. This tests whether probability *shape* (soft
     distribution) is causing noisy rendering. It approximates the "best class"
     view used in PLY visualization.

2) scale:
   - Multiply Gaussian scales by a factor (>1) to test whether sparsity is
     causing speckled renderings. This only affects rendering during eval.

NOTE:
- Top-k class IDs (seman_cls_ids) are unchanged.
- Checkpoints on disk are untouched.
"""

import argparse
import math
import os
import sys

import torch
from tensorboardX import SummaryWriter

# Ensure local imports work the same way as the main entry points.
sys.path.append(os.getcwd())

from src.naruto.cfg_loader import load_cfg
from src.utils.general_utils import fix_random_seed, InfoPrinter
from src.slam import init_SLAM_model


def _log_stats(tag: str, t: torch.Tensor) -> None:
    """Print simple tensor stats to help compare before/after modifications."""
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
            "[eval_semantic_ablation] "
            f"{tag}: bad={bad} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic rendering with ablations (onehot/scale)."
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
        help="Override result directory (required to locate params*.npz)",
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
        "--mode",
        type=str,
        required=True,
        choices=("onehot", "scale", "softmax"),
        help="Ablation mode: onehot, scale, or softmax",
    )
    parser.add_argument(
        "--scale_mul",
        type=float,
        default=2.0,
        help="Scale multiplier for mode=scale (e.g., 2.0 doubles size).",
    )
    parser.add_argument(
        "--out_suffix",
        type=str,
        default="final_ablation",
        help="Suffix for eval output dir (e.g., eval_final_onehot).",
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

    if args.mode == "onehot":
        sem = slam.params.get("semantic_logits", None)
        if sem is None:
            raise RuntimeError("semantic_logits not found in loaded params")
        _log_stats("semantic_logits (raw)", sem)
        # Convert each Gaussian's top-k logits to a one-hot vector.
        idx = torch.argmax(sem, dim=1)
        onehot = torch.zeros_like(sem)
        onehot[torch.arange(sem.shape[0], device=sem.device), idx] = 1.0
        _log_stats("semantic_logits (onehot)", onehot)
        slam.params["semantic_logits"] = torch.nn.Parameter(onehot)

    elif args.mode == "softmax":
        sem = slam.params.get("semantic_logits", None)
        if sem is None:
            raise RuntimeError("semantic_logits not found in loaded params")
        _log_stats("semantic_logits (raw)", sem)
        # Convert each Gaussian's top-k values to a proper probability distribution.
        # This treats the stored values as logits (not necessarily normalized).
        sem_soft = torch.softmax(sem, dim=1)
        _log_stats("semantic_logits (softmax)", sem_soft)
        slam.params["semantic_logits"] = torch.nn.Parameter(sem_soft)

    elif args.mode == "scale":
        log_scales = slam.params.get("log_scales", None)
        if log_scales is None:
            raise RuntimeError("log_scales not found in loaded params")
        _log_stats("log_scales (raw)", log_scales)
        if args.scale_mul <= 0:
            raise ValueError("--scale_mul must be > 0")
        # Multiply scales by factor => add ln(factor) in log-space.
        log_scales_scaled = log_scales + math.log(args.scale_mul)
        _log_stats("log_scales (scaled)", log_scales_scaled)
        slam.params["log_scales"] = torch.nn.Parameter(log_scales_scaled)

    # Run evaluation; results go to eval_<out_suffix> directory.
    slam.eval_semantic_result(
        eval_dir_suffix=args.out_suffix,
        ignore_first_frame=True,
        save_frames=bool(args.save_frames),
    )


if __name__ == "__main__":
    main()
