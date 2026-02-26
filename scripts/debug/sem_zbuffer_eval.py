#!/usr/bin/env python3
"""
Z-buffer semantic evaluation (debug only).

Approximates semantic rendering by projecting Gaussians to pixels and
selecting the nearest (smallest z) Gaussian per pixel, then assigning
its top-1 semantic class. This isolates whether alpha/T blending is
the main source of noisy semantic outputs.

This does NOT modify checkpoints on disk.
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

sys.path.append(os.getcwd())

from src.naruto.cfg_loader import load_cfg
from src.utils.general_utils import fix_random_seed, InfoPrinter
from src.slam import init_SLAM_model
from src.slam.splatam.eval_helper import transform_to_frame
from src.slam.semsplatam.modified_ver.splatam.eval_helper import calc_miou


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z-buffer semantic eval (debug).")
    parser.add_argument("--cfg", type=str, required=True, help="Config file")
    parser.add_argument("--result_dir", type=str, required=True, help="Result dir for params*.npz")
    parser.add_argument("--stage", type=str, default="final", help="Checkpoint stage")
    parser.add_argument("--step", type=int, default=1100, help="Checkpoint step")
    parser.add_argument("--max_frames", type=int, default=20, help="Max evaluated frames")
    parser.add_argument("--max_gaussians", type=int, default=300000, help="Max gaussians sampled per frame")
    parser.add_argument("--downscale", type=int, default=4, help="Downscale factor for eval (>=1)")
    parser.add_argument("--min_opacity", type=float, default=0.0, help="Min opacity to keep gaussians")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Min semantic confidence to keep gaussians")
    parser.add_argument("--out_suffix", type=str, default="final_zbuffer", help="Eval dir suffix")
    return parser.parse_args()


def _select_top1_class(sem_logits: torch.Tensor, cls_ids: torch.Tensor) -> torch.Tensor:
    """
    sem_logits: (N, K)
    cls_ids: (N, K) class ids for each gaussian top-k slot
    Returns per-gaussian class id: (N,)
    """
    top1_idx = sem_logits.argmax(dim=1)
    row = torch.arange(sem_logits.shape[0], device=sem_logits.device)
    return cls_ids[row, top1_idx]


@torch.no_grad()
def _eval_frame(means_cam: torch.Tensor, cls_id: torch.Tensor, intrinsics: torch.Tensor,
                gt_sem: torch.Tensor, downscale: int) -> float:
    """
    means_cam: (N,3) in camera coords
    cls_id: (N,) class id per gaussian
    intrinsics: (3,3)
    gt_sem: (H,W) long
    downscale: int
    Returns mIoU for this frame.
    """
    H, W = gt_sem.shape
    if downscale > 1:
        Hs, Ws = H // downscale, W // downscale
        gt_sem = F.interpolate(gt_sem[None, None].float(), size=(Hs, Ws), mode="nearest")[0, 0].long()
    else:
        Hs, Ws = H, W

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if downscale > 1:
        fx = fx / downscale
        fy = fy / downscale
        cx = cx / downscale
        cy = cy / downscale

    z = means_cam[:, 2]
    valid = z > 1e-6
    if valid.sum() == 0:
        return 0.0
    means = means_cam[valid]
    cls = cls_id[valid]
    z = z[valid]

    x = means[:, 0] / z
    y = means[:, 1] / z
    u = (fx * x + cx).round().long()
    v = (fy * y + cy).round().long()

    in_bounds = (u >= 0) & (u < Ws) & (v >= 0) & (v < Hs)
    if in_bounds.sum() == 0:
        return 0.0
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    cls = cls[in_bounds]

    idx = v * Ws + u
    # Min-depth per pixel
    inf = torch.tensor(float("inf"), device=z.device)
    min_z = torch.full((Hs * Ws,), inf, device=z.device)
    min_z.scatter_reduce_(0, idx, z, reduce="amin")

    # Keep gaussians at min depth (ties resolved by last write)
    mask = z == min_z[idx]
    idx = idx[mask]
    cls = cls[mask]

    pred = torch.zeros((Hs * Ws,), device=z.device, dtype=torch.long)
    pred.scatter_(0, idx, cls)
    pred = pred.view(Hs, Ws)

    return calc_miou(pred=pred, target=gt_sem)


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
    cls_ids = slam.variables["seman_cls_ids"].to(slam.device)
    sem_logits = slam.params["semantic_logits"].detach()
    opacities = torch.sigmoid(slam.params["logit_opacities"].detach().squeeze(1))
    top1_cls = _select_top1_class(sem_logits, cls_ids)
    # Semantic confidence as max over positive-normalized logits (top-k only).
    sem_pos = torch.clamp(sem_logits, min=0.0)
    sem_sum = sem_pos.sum(dim=1)
    sem_sum = torch.where(sem_sum == 0, torch.ones_like(sem_sum), sem_sum)
    sem_conf = (sem_pos.max(dim=1).values / sem_sum)

    eval_every = slam.config.get("eval_every", 1)
    miou_list: List[float] = []

    for time_idx in range(len(dataset)):
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue
        if len(miou_list) >= args.max_frames:
            break

        _, _, intrinsics, pose = dataset[time_idx]
        intrinsics = intrinsics[:3, :3].to(slam.device)
        gt_w2c = torch.linalg.inv(pose)
        gt_sem = dataset.get_semantic_map(time_idx)[0].to(slam.device)

        transformed = transform_to_frame(
            slam.params,
            time_idx,
            gaussians_grad=False,
            camera_grad=False,
            rel_w2c=gt_w2c,
        )
        means = transformed["means3D"]

        # Subsample gaussians if requested
        N = means.shape[0]
        keep = (opacities >= args.min_opacity) & (sem_conf >= args.min_conf)
        if keep.sum() == 0:
            continue
        means = means[keep]
        cls = top1_cls[keep]
        N = means.shape[0]
        if N > args.max_gaussians:
            perm = torch.randperm(N, device=means.device)[: args.max_gaussians]
            means = means[perm]
            cls = cls[perm]
        else:
            cls = cls

        miou = _eval_frame(means, cls, intrinsics, gt_sem, args.downscale)
        miou_list.append(miou)

    eval_dir = os.path.join(slam.config["workdir"], f"eval_{args.out_suffix}")
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "zbuffer_miou.txt")
    with open(out_path, "w") as f:
        f.write(f"stage: {args.stage}\nstep: {args.step}\n")
        f.write(f"frames: {len(miou_list)}\n")
        if miou_list:
            f.write(f"miou_mean: {float(np.mean(miou_list)):.6f}\n")
            f.write(f"miou_median: {float(np.median(miou_list)):.6f}\n")
            f.write(f"miou_min: {float(np.min(miou_list)):.6f}\n")
            f.write(f"miou_max: {float(np.max(miou_list)):.6f}\n")
        else:
            f.write("miou_mean: 0.0\n")

    print(f"[sem_zbuffer_eval] wrote {out_path}")


if __name__ == "__main__":
    main()
