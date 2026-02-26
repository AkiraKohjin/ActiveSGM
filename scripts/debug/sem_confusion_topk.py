#!/usr/bin/env python3
"""
Semantic evaluation diagnostics: confusion matrix + top-k recall.

This script does NOT change checkpoints on disk. It loads a checkpoint and
modifies in-memory semantics only if requested (e.g., onehot).

Use cases:
- Confusion analysis with one-hot rendering (best-class view).
- Top-k recall analysis with raw/soft distributions.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from tensorboardX import SummaryWriter

sys.path.append(os.getcwd())

from src.naruto.cfg_loader import load_cfg
from src.utils.general_utils import fix_random_seed, InfoPrinter
from src.slam import init_SLAM_model

from src.slam.splatam.eval_helper import transform_to_frame
from src.slam.semsplatam.modified_ver.splatam.splatam import (
    setup_camera,
    set_camera_sparse,
    transformed_params2rendervar,
    transformed_params2semrendervar_sparse,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from sparse_channel_rasterization import GaussianRasterizer as SEMRenderer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic confusion matrix + top-k recall analysis."
    )
    parser.add_argument("--cfg", type=str, required=True, help="Config file")
    parser.add_argument("--result_dir", type=str, required=True, help="Result dir for params*.npz")
    parser.add_argument("--stage", type=str, default="final", help="Checkpoint stage")
    parser.add_argument("--step", type=int, default=1100, help="Checkpoint step")
    parser.add_argument(
        "--mode",
        type=str,
        default="raw",
        choices=("raw", "posnorm", "softmax", "onehot"),
        help="How to treat semantic_logits before rendering",
    )
    parser.add_argument("--out_suffix", type=str, default="final_confusion", help="Eval dir suffix")
    parser.add_argument("--save_frames", type=int, default=0, help="Save per-frame debug renders")
    parser.add_argument("--topk", type=int, default=16, help="Top-k for recall")
    parser.add_argument("--downscale", type=int, default=1, help="Downscale factor (>=1)")
    parser.add_argument("--max_frames", type=int, default=-1, help="Max evaluated frames (-1 = all)")
    return parser.parse_args()


def _load_class_names(semantic_dir: str, n_cls: int) -> Dict[int, str]:
    """Load Replica class names from info_semantic.json if available."""
    info_path = os.path.join(semantic_dir, "info_semantic.json")
    if not os.path.exists(info_path):
        return {i: f"class_{i}" for i in range(n_cls)}
    with open(info_path, "r") as f:
        info = json.load(f)
    names = {0: "unknown"}
    for cls in info.get("classes", []):
        names[int(cls["id"])] = cls["name"]
    # Fill any gaps
    for i in range(n_cls):
        if i not in names:
            names[i] = f"class_{i}"
    return names


def _apply_mode(sem: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return sem
    if mode == "posnorm":
        sem = torch.clamp(sem, min=0)
        denom = sem.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return sem / denom
    if mode == "softmax":
        return torch.softmax(sem, dim=1)
    if mode == "onehot":
        idx = torch.argmax(sem, dim=1)
        out = torch.zeros_like(sem)
        out[torch.arange(sem.shape[0], device=sem.device), idx] = 1.0
        return out
    raise ValueError(f"Unknown mode {mode}")


def _update_confusion(conf: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> None:
    """Accumulate confusion matrix for non-zero GT pixels."""
    mask = gt != 0
    if mask.sum() == 0:
        return
    gt_m = gt[mask].astype(np.int64)
    pr_m = pred[mask].astype(np.int64)
    # Vectorized bincount on flattened pairs.
    n = conf.shape[0]
    inds = gt_m * n + pr_m
    binc = np.bincount(inds, minlength=n * n)
    conf += binc.reshape(n, n)


def _topk_recall(pred_logits: torch.Tensor, target: torch.Tensor, k: int) -> float:
    """Top-k recall for GT classes (ignore GT==0)."""
    H, W, C = pred_logits.shape
    pred_flat = pred_logits.reshape(-1, C)
    tgt_flat = target.view(-1).to(pred_flat.device)
    mask = tgt_flat != 0
    if mask.sum() == 0:
        return 0.0
    pred_flat = pred_flat[mask]
    tgt_flat = tgt_flat[mask]
    _, topk = pred_flat.topk(k, dim=1, largest=True, sorted=True)
    hit = topk.eq(tgt_flat.unsqueeze(1)).any(dim=1).float().mean().item()
    return hit


def main() -> None:
    args = _parse_args()
    info_printer = InfoPrinter("ActiveSem")
    main_cfg = load_cfg(args)
    fix_random_seed(main_cfg.general.seed)

    # Logger (same pattern as other eval scripts)
    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    _ = SummaryWriter(f"{log_savedir}")

    slam = init_SLAM_model(main_cfg, info_printer, None)
    slam.load_params_by_step(step=args.step, stage=args.stage)

    # Apply semantic_logits mode in-memory.
    sem = slam.params.get("semantic_logits", None)
    if sem is None:
        raise RuntimeError("semantic_logits not found in params")
    sem_mod = _apply_mode(sem, args.mode)
    slam.params["semantic_logits"] = torch.nn.Parameter(sem_mod)

    dataset = slam.dataset_eval
    num_frames = len(dataset)
    n_cls = slam.n_cls
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)

    # Class names (Replica)
    semantic_dir = getattr(main_cfg.slam, "semantic_dir", "")
    class_names = _load_class_names(semantic_dir, n_cls)

    topk_list = [1, 3, 5, args.topk]
    topk_accum = {k: [] for k in topk_list}

    frame_count = 0
    for time_idx in range(num_frames):
        color, _, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        intrinsics = intrinsics[:3, :3]
        seman_gt = dataset.get_semantic_map(time_idx)[0]

        if time_idx == 0:
            first_frame_w2c = torch.linalg.inv(pose)
            cam = setup_camera(
                color.shape[1],
                color.shape[0],
                intrinsics.cpu().numpy(),
                first_frame_w2c.detach().cpu().numpy(),
                num_channels=n_cls,
            )

        # Match eval_every behavior
        eval_every = slam.config.get("eval_every", 1)
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue

        transformed_gaussians = transform_to_frame(
            slam.params,
            time_idx,
            gaussians_grad=False,
            camera_grad=False,
            rel_w2c=gt_w2c,
        )

        # Compute seen mask
        rendervar = transformed_params2rendervar(slam.params, transformed_gaussians)
        _, radius, _ = Renderer(raster_settings=cam)(**rendervar)
        seen = radius > 0

        # Semantic render
        sem_cam = cam
        if "seman_cls_ids" in slam.variables:
            cls_ids = slam.variables["seman_cls_ids"].to(seen.device)
            if cls_ids.shape[0] == seen.shape[0]:
                cls_ids = cls_ids[seen]
            sem_cam = set_camera_sparse(cam=cam, cls_ids=cls_ids)

        seman_rendervar = transformed_params2semrendervar_sparse(
            slam.params, transformed_gaussians, seen
        )
        rastered_seman, _ = SEMRenderer(raster_settings=sem_cam)(**seman_rendervar)
        rastered_seman = torch.nan_to_num(rastered_seman, nan=0.0)
        rastered_seman[rastered_seman < 0] = 0.0

        rastered_seman = rastered_seman.permute(1, 2, 0)  # H, W, C
        if args.downscale > 1:
            rastered_seman = rastered_seman[:: args.downscale, :: args.downscale, :]
            seman_gt = seman_gt[:: args.downscale, :: args.downscale]
        rastered_cls = rastered_seman.argmax(-1).detach().cpu().numpy()
        gt_np = seman_gt.detach().cpu().numpy()

        _update_confusion(conf, rastered_cls, gt_np)

        # Top-k recall
        for k in topk_list:
            acc = _topk_recall(rastered_seman, seman_gt.long(), k)
            topk_accum[k].append(acc)
        frame_count += 1
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

    # Output directory
    eval_dir = slam.eval_dir + "_" + args.out_suffix
    os.makedirs(eval_dir, exist_ok=True)

    # Save confusion matrix
    np.save(os.path.join(eval_dir, "confusion.npy"), conf)

    # Per-class metrics
    diag = np.diag(conf)
    gt_sum = conf.sum(axis=1)
    pred_sum = conf.sum(axis=0)
    iou = diag / (gt_sum + pred_sum - diag + 1e-9)
    acc = diag / (gt_sum + 1e-9)

    # Write summary
    with open(os.path.join(eval_dir, "confusion_summary.txt"), "w") as f:
        f.write(f"mode: {args.mode}\\n")
        f.write(f"topk: {args.topk}\\n")
        f.write(f"frames: {frame_count}\\n")
        f.write(f"downscale: {args.downscale}\\n")
        f.write("Top-k recall (mean over frames):\\n")
        for k in topk_list:
            vals = topk_accum[k]
            mean_v = float(np.mean(vals)) if vals else 0.0
            f.write(f"  top{k}: {mean_v:.4f}\\n")

        f.write("\nPer-class IoU/Acc (gt != 0):\n")
        for cls_id in range(1, n_cls):
            f.write(
                f"{cls_id:3d} {class_names.get(cls_id,'?'):<24} "
                f"IoU={iou[cls_id]:.4f} Acc={acc[cls_id]:.4f} gt={gt_sum[cls_id]}\n"
            )

        f.write("\nTop confusions per GT class:\n")
        for cls_id in range(1, n_cls):
            row = conf[cls_id].copy()
            row[cls_id] = 0
            if row.sum() == 0:
                continue
            top_idx = row.argsort()[-5:][::-1]
            f.write(f"GT {cls_id:3d} {class_names.get(cls_id,'?')}: ")
            pairs = []
            for j in top_idx:
                if row[j] == 0:
                    continue
                pairs.append(f"{j}({class_names.get(j,'?')})={row[j]}")
            f.write(", ".join(pairs) + "\n")


if __name__ == "__main__":
    main()
