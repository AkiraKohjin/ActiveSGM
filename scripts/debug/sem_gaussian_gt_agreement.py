#!/usr/bin/env python3
"""
Measure agreement between per-Gaussian semantic labels and 2D GT after projection.

Goal:
- Determine whether semantics are wrong at the Gaussian level (before rendering).
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from tensorboardX import SummaryWriter

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "third_parties", "splatam"))

from src.naruto.cfg_loader import load_cfg
from src.utils.general_utils import fix_random_seed, InfoPrinter
from src.slam import init_SLAM_model

from src.slam.semsplatam.modified_ver.splatam.eval_helper import transform_to_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian vs GT semantic agreement")
    parser.add_argument("--cfg", type=str, required=True, help="Config file")
    parser.add_argument("--result_dir", type=str, required=True, help="Result dir for params*.npz")
    parser.add_argument("--stage", type=str, default="final", help="Checkpoint stage")
    parser.add_argument("--step", type=int, default=1100, help="Checkpoint step")
    parser.add_argument("--max_frames", type=int, default=30, help="Max evaluated frames")
    parser.add_argument("--max_gaussians", type=int, default=200000, help="Max gaussians sampled per frame")
    parser.add_argument("--out_suffix", type=str, default="final_gaussian_gt", help="Eval dir suffix")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    info_printer = InfoPrinter("ActiveSem")
    main_cfg = load_cfg(args)
    fix_random_seed(main_cfg.general.seed)

    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    _ = SummaryWriter(f"{log_savedir}")

    slam = init_SLAM_model(main_cfg, info_printer, None)
    slam.load_params_by_step(step=args.step, stage=args.stage)

    dataset = slam.dataset_eval
    n_cls = slam.n_cls
    sem_logits = slam.params["semantic_logits"].detach()
    cls_ids = slam.variables["seman_cls_ids"].detach()

    # Per-Gaussian predicted class (top1 within TOPK)
    topk_idx = sem_logits.argmax(dim=1)
    gauss_cls = cls_ids[torch.arange(cls_ids.shape[0], device=cls_ids.device), topk_idx]

    eval_every = slam.config.get("eval_every", 1)
    frame_acc: List[float] = []
    frame_count = 0

    for time_idx in range(len(dataset)):
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue
        if frame_count >= args.max_frames:
            break

        color, _, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        intrinsics = intrinsics[:3, :3].to(slam.device)
        seman_gt = dataset.get_semantic_map(time_idx)[0].to(slam.device)

        transformed = transform_to_frame(
            slam.params,
            time_idx,
            gaussians_grad=False,
            camera_grad=False,
            rel_w2c=gt_w2c,
        )
        means = transformed["means3D"]  # (N,3) in camera coords

        # Optional sampling to cap cost
        N = means.shape[0]
        if N > args.max_gaussians:
            perm = torch.randperm(N, device=means.device)[: args.max_gaussians]
            means = means[perm]
            g_cls = gauss_cls[perm]
        else:
            g_cls = gauss_cls

        z = means[:, 2]
        valid = z > 1e-6
        means = means[valid]
        g_cls = g_cls[valid]

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        x = means[:, 0] / z[valid]
        y = means[:, 1] / z[valid]
        u = fx * x + cx
        v = fy * y + cy

        H, W = seman_gt.shape
        u = u.round().long()
        v = v.round().long()
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if in_bounds.sum() == 0:
            continue

        u = u[in_bounds]
        v = v[in_bounds]
        g_cls = g_cls[in_bounds]
        gt_cls = seman_gt[v, u]

        # Ignore GT==0
        mask = gt_cls != 0
        if mask.sum() == 0:
            continue

        acc = (g_cls[mask] == gt_cls[mask]).float().mean().item()
        frame_acc.append(acc)
        frame_count += 1

    eval_dir = os.path.join(slam.config["workdir"], f"eval_{args.out_suffix}")
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "gaussian_gt_summary.txt")
    with open(out_path, "w") as f:
        if frame_acc:
            f.write(f"frames: {len(frame_acc)}\n")
            f.write(f"acc_mean: {float(np.mean(frame_acc)):.4f}\n")
            f.write(f"acc_median: {float(np.median(frame_acc)):.4f}\n")
            f.write(f"acc_min: {float(np.min(frame_acc)):.4f}\n")
            f.write(f"acc_max: {float(np.max(frame_acc)):.4f}\n")
        else:
            f.write("frames: 0\nacc_mean: 0.0\n")

    print(f"[gaussian_gt] wrote {out_path}")


if __name__ == "__main__":
    main()
