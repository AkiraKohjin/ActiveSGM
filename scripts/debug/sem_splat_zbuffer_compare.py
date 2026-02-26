#!/usr/bin/env python3
"""
Compare semantic rendering vs z-buffer-with-splat approximation.

Goal:
- Approximate splat footprint using renderer radii (no alpha/T blending).
- Compare per-frame mIoU and class distribution between:
    (a) SEMRenderer with one-hot per-Gaussian semantics
    (b) Z-buffer with splat radius (square footprint)

This isolates whether the semantic renderer is the primary source of noise.
"""

import argparse
import os
import sys
from typing import Dict, List

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
from src.slam.semsplatam.modified_ver.splatam.splatam import (
    setup_camera,
    set_camera_sparse,
    transformed_params2rendervar,
    transformed_params2semrendervar_sparse,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from sparse_channel_rasterization import GaussianRasterizer as SEMRenderer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare renderer vs splat z-buffer (debug).")
    parser.add_argument("--cfg", type=str, required=True, help="Config file")
    parser.add_argument("--result_dir", type=str, required=True, help="Result dir for params*.npz")
    parser.add_argument("--stage", type=str, default="final", help="Checkpoint stage")
    parser.add_argument("--step", type=int, default=1100, help="Checkpoint step")
    parser.add_argument("--max_frames", type=int, default=10, help="Max evaluated frames")
    parser.add_argument("--max_gaussians", type=int, default=200000, help="Max gaussians sampled per frame")
    parser.add_argument("--downscale", type=int, default=8, help="Downscale factor for z-buffer grid (>=1)")
    parser.add_argument("--min_opacity", type=float, default=0.0, help="Min opacity to keep gaussians")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Min semantic confidence to keep gaussians")
    parser.add_argument("--out_suffix", type=str, default="final_splat_zbuffer_compare", help="Eval dir suffix")
    return parser.parse_args()


def _top1_class(sem_logits: torch.Tensor, cls_ids: torch.Tensor) -> torch.Tensor:
    top1_idx = sem_logits.argmax(dim=1)
    row = torch.arange(sem_logits.shape[0], device=sem_logits.device)
    return cls_ids[row, top1_idx]


def _semantic_confidence(sem_logits: torch.Tensor) -> torch.Tensor:
    sem_pos = torch.clamp(sem_logits, min=0.0)
    sem_sum = sem_pos.sum(dim=1)
    sem_sum = torch.where(sem_sum == 0, torch.ones_like(sem_sum), sem_sum)
    return sem_pos.max(dim=1).values / sem_sum


def _zbuffer_splat(
    means_cam: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    radii: np.ndarray,
    intrinsics: np.ndarray,
    gt_sem: np.ndarray,
    downscale: int,
) -> np.ndarray:
    """Return predicted class map (Hs, Ws) using z-buffer with square splat footprint."""
    H, W = gt_sem.shape
    if downscale > 1:
        Hs, Ws = H // downscale, W // downscale
        gt_sem = gt_sem[: Hs * downscale, : Ws * downscale]
    else:
        Hs, Ws = H, W

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if downscale > 1:
        fx = fx / downscale
        fy = fy / downscale
        cx = cx / downscale
        cy = cy / downscale

    x = means_cam[:, 0] / z
    y = means_cam[:, 1] / z
    u = np.rint(fx * x + cx).astype(np.int32)
    v = np.rint(fy * y + cy).astype(np.int32)

    # Downscale radii to match target grid
    r = np.maximum(1, np.rint(radii / float(downscale)).astype(np.int32))

    zbuf = np.full((Hs, Ws), np.inf, dtype=np.float32)
    clsbuf = np.zeros((Hs, Ws), dtype=np.int32)

    for i in range(u.shape[0]):
        ui = u[i]
        vi = v[i]
        if ui < 0 or ui >= Ws or vi < 0 or vi >= Hs:
            continue
        ri = r[i]
        u0 = max(0, ui - ri)
        u1 = min(Ws - 1, ui + ri)
        v0 = max(0, vi - ri)
        v1 = min(Hs - 1, vi + ri)

        # Square footprint update
        z_slice = zbuf[v0 : v1 + 1, u0 : u1 + 1]
        mask = z[i] < z_slice
        if not np.any(mask):
            continue
        z_slice[mask] = z[i]
        cls_slice = clsbuf[v0 : v1 + 1, u0 : u1 + 1]
        cls_slice[mask] = cls[i]

    return clsbuf


def _class_hist(pred: np.ndarray, valid_mask: np.ndarray, n_cls: int) -> np.ndarray:
    ids = pred[valid_mask].reshape(-1)
    if ids.size == 0:
        return np.zeros((n_cls,), dtype=np.float32)
    hist = np.bincount(ids, minlength=n_cls).astype(np.float32)
    hist /= max(1, hist.sum())
    return hist


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
    cls_ids = slam.variables["seman_cls_ids"].to(slam.device)
    top1_cls = _top1_class(sem_logits, cls_ids)
    sem_conf = _semantic_confidence(sem_logits)
    opacities = torch.sigmoid(slam.params["logit_opacities"].detach().squeeze(1))

    eval_every = slam.config.get("eval_every", 1)
    frame_count = 0

    miou_renderer: List[float] = []
    miou_zbuf: List[float] = []
    hist_l1: List[float] = []

    for time_idx in range(len(dataset)):
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue
        if frame_count >= args.max_frames:
            break

        color, _, intrinsics, pose = dataset[time_idx]
        intrinsics = intrinsics[:3, :3]
        gt_w2c = torch.linalg.inv(pose)
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

        transformed = transform_to_frame(
            slam.params,
            time_idx,
            gaussians_grad=False,
            camera_grad=False,
            rel_w2c=gt_w2c,
        )

        # Compute radii via RGB renderer (no semantic renderer involvement)
        rendervar = transformed_params2rendervar(slam.params, transformed)
        _, radius, _ = Renderer(raster_settings=cam)(**rendervar)
        seen = radius > 0

        # Build onehot semantics for renderer
        sem_logits_frame = sem_logits[seen]
        top1_idx = sem_logits_frame.argmax(dim=1)
        onehot = torch.zeros_like(sem_logits_frame)
        onehot[torch.arange(onehot.shape[0], device=onehot.device), top1_idx] = 1.0

        sem_cam = cam
        if "seman_cls_ids" in slam.variables:
            cls_ids_frame = slam.variables["seman_cls_ids"].to(seen.device)
            if cls_ids_frame.shape[0] == seen.shape[0]:
                cls_ids_frame = cls_ids_frame[seen]
            sem_cam = set_camera_sparse(cam=cam, cls_ids=cls_ids_frame)

        seman_rendervar = transformed_params2semrendervar_sparse(
            slam.params, transformed, seen
        )
        seman_rendervar["colors_precomp"] = onehot
        rastered_onehot, _ = SEMRenderer(raster_settings=sem_cam)(**seman_rendervar)

        rastered_onehot = torch.nan_to_num(rastered_onehot, nan=0.0)
        rastered_onehot[rastered_onehot < 0] = 0.0
        pred_renderer = rastered_onehot.argmax(dim=0).detach().cpu().numpy()

        # Z-buffer with splat radius (square footprint)
        means = transformed["means3D"][seen]
        z = means[:, 2]
        valid = z > 1e-6
        means = means[valid]
        z = z[valid]
        radius_f = radius[seen][valid]

        # Apply filters
        cls_all = top1_cls[seen][valid]
        keep = (opacities[seen][valid] >= args.min_opacity) & (sem_conf[seen][valid] >= args.min_conf)
        if keep.sum() == 0:
            continue
        means = means[keep]
        z = z[keep]
        radius_f = radius_f[keep]
        cls_all = cls_all[keep]

        # Subsample gaussians
        if means.shape[0] > args.max_gaussians:
            perm = torch.randperm(means.shape[0], device=means.device)[: args.max_gaussians]
            means = means[perm]
            z = z[perm]
            radius_f = radius_f[perm]
            cls_all = cls_all[perm]

        # Move to CPU numpy for z-buffer
        means_np = means.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        cls_np = cls_all.detach().cpu().numpy()
        radius_np = radius_f.detach().cpu().numpy()
        intr_np = intrinsics.detach().cpu().numpy()
        gt_np = seman_gt.detach().cpu().numpy()

        pred_zbuf = _zbuffer_splat(
            means_np, z_np, cls_np, radius_np, intr_np, gt_np, args.downscale
        )

        # Compute mIoU on downscaled GT
        if args.downscale > 1:
            Hs = gt_np.shape[0] // args.downscale
            Ws = gt_np.shape[1] // args.downscale
            gt_ds = gt_np[: Hs * args.downscale, : Ws * args.downscale]
            gt_ds = gt_ds.reshape(Hs, args.downscale, Ws, args.downscale)
            gt_ds = gt_ds[:, 0, :, 0]  # nearest subsample
        else:
            gt_ds = gt_np

        miou_z = calc_miou(
            torch.from_numpy(pred_zbuf).contiguous(),
            torch.from_numpy(gt_ds).contiguous(),
        )

        # Downscale renderer prediction and GT for fair comparison
        if args.downscale > 1:
            pred_r_ds = pred_renderer[:: args.downscale, :: args.downscale]
            gt_r_ds = gt_ds
        else:
            pred_r_ds = pred_renderer
            gt_r_ds = gt_np
        miou_renderer.append(
            calc_miou(
                torch.from_numpy(pred_r_ds).contiguous(),
                torch.from_numpy(gt_r_ds).contiguous(),
            )
        )
        miou_zbuf.append(miou_z)

        valid_mask = gt_ds != 0
        hist_r = _class_hist(pred_renderer[:: args.downscale, :: args.downscale], valid_mask, n_cls)
        hist_z = _class_hist(pred_zbuf, valid_mask, n_cls)
        hist_l1.append(float(np.abs(hist_r - hist_z).sum()))

        frame_count += 1

    eval_dir = os.path.join(slam.config["workdir"], f"eval_{args.out_suffix}")
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "compare_summary.txt")
    with open(out_path, "w") as f:
        f.write(f"stage: {args.stage}\nstep: {args.step}\nframes: {frame_count}\n")
        if miou_renderer:
            f.write(f"miou_renderer_mean: {float(np.mean(miou_renderer)):.6f}\n")
            f.write(f"miou_zbuf_mean: {float(np.mean(miou_zbuf)):.6f}\n")
            f.write(f"hist_l1_mean: {float(np.mean(hist_l1)):.6f}\n")
        else:
            f.write("miou_renderer_mean: 0.0\nmiou_zbuf_mean: 0.0\nhist_l1_mean: 0.0\n")

    print(f"[sem_splat_zbuffer_compare] wrote {out_path}")


if __name__ == "__main__":
    main()
