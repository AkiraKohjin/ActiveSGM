#!/usr/bin/env python3
"""
Per-class IoU/Acc for Gaussian semantics via z-buffer projection.

This approximates per-pixel prediction by projecting Gaussians to the image
plane and selecting the nearest (smallest z) Gaussian per pixel.
No renderer is used.
"""

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import torch
from tensorboardX import SummaryWriter

sys.path.append(os.getcwd())

from src.naruto.cfg_loader import load_cfg
from src.utils.general_utils import fix_random_seed, InfoPrinter
from src.slam import init_SLAM_model
from src.slam.splatam.eval_helper import transform_to_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian z-buffer confusion.")
    parser.add_argument("--cfg", type=str, required=True, help="Config file")
    parser.add_argument("--result_dir", type=str, required=True, help="Result dir")
    parser.add_argument("--stage", type=str, default="final", help="Checkpoint stage")
    parser.add_argument("--step", type=int, default=1100, help="Checkpoint step")
    parser.add_argument("--max_frames", type=int, default=20, help="Max evaluated frames")
    parser.add_argument("--max_gaussians", type=int, default=200000, help="Max gaussians sampled per frame")
    parser.add_argument("--downscale", type=int, default=4, help="Downscale factor (>=1)")
    parser.add_argument("--min_opacity", type=float, default=0.0, help="Min opacity to keep gaussians")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Min semantic confidence to keep gaussians")
    parser.add_argument("--out_suffix", type=str, default="final_gaussian_zbuf_confusion", help="Eval dir suffix")
    return parser.parse_args()


def _load_class_names(semantic_dir: str, n_cls: int) -> Dict[int, str]:
    info_path = os.path.join(semantic_dir, "info_semantic.json")
    if not os.path.exists(info_path):
        return {i: f"class_{i}" for i in range(n_cls)}
    with open(info_path, "r") as f:
        info = json.load(f)
    names = {0: "unknown"}
    for cls in info.get("classes", []):
        names[int(cls["id"])] = cls["name"]
    for i in range(n_cls):
        if i not in names:
            names[i] = f"class_{i}"
    return names


def _update_confusion(conf: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> None:
    mask = gt != 0
    if mask.sum() == 0:
        return
    gt_m = gt[mask].astype(np.int64)
    pr_m = pred[mask].astype(np.int64)
    n = conf.shape[0]
    inds = gt_m * n + pr_m
    binc = np.bincount(inds, minlength=n * n)
    conf += binc.reshape(n, n)


def _zbuffer_pred(
    means: torch.Tensor,
    cls: torch.Tensor,
    intrinsics: torch.Tensor,
    H: int,
    W: int,
    downscale: int,
) -> torch.Tensor:
    if downscale > 1:
        Hs, Ws = H // downscale, W // downscale
        fx = intrinsics[0, 0] / downscale
        fy = intrinsics[1, 1] / downscale
        cx = intrinsics[0, 2] / downscale
        cy = intrinsics[1, 2] / downscale
    else:
        Hs, Ws = H, W
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = means[:, 2]
    valid = z > 1e-6
    means = means[valid]
    cls = cls[valid]
    z = z[valid]

    x = means[:, 0] / z
    y = means[:, 1] / z
    u = (fx * x + cx).round().long()
    v = (fy * y + cy).round().long()

    in_bounds = (u >= 0) & (u < Ws) & (v >= 0) & (v < Hs)
    if in_bounds.sum() == 0:
        return torch.zeros((Hs, Ws), device=means.device, dtype=torch.long)
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    cls = cls[in_bounds]

    idx = v * Ws + u
    inf = torch.tensor(float("inf"), device=means.device)
    min_z = torch.full((Hs * Ws,), inf, device=means.device)
    min_z.scatter_reduce_(0, idx, z, reduce="amin")

    mask = z == min_z[idx]
    idx = idx[mask]
    cls = cls[mask]

    pred = torch.zeros((Hs * Ws,), device=means.device, dtype=torch.long)
    pred.scatter_(0, idx, cls)
    return pred.view(Hs, Ws)


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
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)

    semantic_dir = getattr(main_cfg.slam, "semantic_dir", "")
    class_names = _load_class_names(semantic_dir, n_cls)

    sem_logits = slam.params["semantic_logits"].detach()
    cls_ids = slam.variables["seman_cls_ids"].detach()
    top1_idx = sem_logits.argmax(dim=1)
    top1_cls = cls_ids[torch.arange(cls_ids.shape[0], device=cls_ids.device), top1_idx]
    sem_pos = torch.clamp(sem_logits, min=0.0)
    sem_sum = sem_pos.sum(dim=1)
    sem_sum = torch.where(sem_sum == 0, torch.ones_like(sem_sum), sem_sum)
    sem_conf = sem_pos.max(dim=1).values / sem_sum
    opacities = torch.sigmoid(slam.params["logit_opacities"].detach().squeeze(1))

    eval_every = slam.config.get("eval_every", 1)
    frame_count = 0

    for time_idx in range(len(dataset)):
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue
        if frame_count >= args.max_frames:
            break

        _, _, intrinsics, pose = dataset[time_idx]
        intrinsics = intrinsics[:3, :3].to(slam.device)
        gt = dataset.get_semantic_map(time_idx)[0].to(slam.device)
        gt_w2c = torch.linalg.inv(pose)

        transformed = transform_to_frame(
            slam.params,
            time_idx,
            gaussians_grad=False,
            camera_grad=False,
            rel_w2c=gt_w2c,
        )
        means = transformed["means3D"]

        keep = (opacities >= args.min_opacity) & (sem_conf >= args.min_conf)
        means = means[keep]
        cls = top1_cls[keep]

        if means.shape[0] == 0:
            continue

        if means.shape[0] > args.max_gaussians:
            perm = torch.randperm(means.shape[0], device=means.device)[: args.max_gaussians]
            means = means[perm]
            cls = cls[perm]

        pred = _zbuffer_pred(
            means, cls, intrinsics, gt.shape[0], gt.shape[1], args.downscale
        )

        if args.downscale > 1:
            gt = gt[:: args.downscale, :: args.downscale]

        _update_confusion(
            conf,
            pred.detach().cpu().numpy(),
            gt.detach().cpu().numpy(),
        )
        frame_count += 1

    eval_dir = slam.eval_dir + "_" + args.out_suffix
    os.makedirs(eval_dir, exist_ok=True)
    np.save(os.path.join(eval_dir, "confusion.npy"), conf)

    diag = np.diag(conf)
    gt_sum = conf.sum(axis=1)
    pred_sum = conf.sum(axis=0)
    iou = diag / (gt_sum + pred_sum - diag + 1e-9)
    acc = diag / (gt_sum + 1e-9)

    classes = [i for i in range(n_cls) if i != 0 and gt_sum[i] > 0]
    miou = float(np.mean([iou[i] for i in classes])) if classes else 0.0

    with open(os.path.join(eval_dir, "confusion_summary.txt"), "w") as f:
        f.write(f"frames: {frame_count}\n")
        f.write(f"miou: {miou:.6f}\n")
        f.write("Per-class IoU/Acc (gt != 0):\n")
        for cid in range(n_cls):
            if cid == 0:
                continue
            f.write(
                f"{cid:3d} {class_names.get(cid, f'class_{cid}'):25s} "
                f"IoU={iou[cid]:.4f} Acc={acc[cid]:.4f} gt={int(gt_sum[cid])}\n"
            )

    print(f"[sem_gaussian_zbuf_confusion] wrote {eval_dir}")


if __name__ == "__main__":
    main()
