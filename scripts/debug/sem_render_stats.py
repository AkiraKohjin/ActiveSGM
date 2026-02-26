#!/usr/bin/env python3
"""
Semantic render diagnostics: measure negative coverage, argmax flip after clamp,
logit mass, and margin statistics for rastered semantics.

This script does NOT change checkpoints on disk.
"""

import argparse
import os
import sys
from typing import Dict, List

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
from src.slam.semsplatam.modified_ver.semantic.oneformer import positive_normalize
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from sparse_channel_rasterization import GaussianRasterizer as SEMRenderer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic render diagnostics.")
    parser.add_argument("--cfg", type=str, required=True, help="Config file")
    parser.add_argument("--result_dir", type=str, required=True, help="Result dir for params*.npz")
    parser.add_argument("--stage", type=str, default="final", help="Checkpoint stage")
    parser.add_argument("--step", type=int, default=1100, help="Checkpoint step")
    parser.add_argument("--max_frames", type=int, default=60, help="Max evaluated frames")
    parser.add_argument("--out_suffix", type=str, default="final_render_stats", help="Eval dir suffix")
    return parser.parse_args()


def _quantiles(x: torch.Tensor, qs: List[float]) -> Dict[str, float]:
    if x.numel() == 0:
        return {f"q{int(q*100)}": 0.0 for q in qs}
    vals = torch.quantile(x, torch.tensor(qs, device=x.device))
    return {f"q{int(q*100)}": float(v.item()) for q, v in zip(qs, vals)}


def _frame_stats(rastered: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    # rastered: C,H,W (float), gt: H,W (long)
    C, H, W = rastered.shape
    logits = rastered.permute(1, 2, 0).reshape(-1, C)
    gt_flat = gt.reshape(-1)
    mask = gt_flat != 0
    if mask.sum() == 0:
        return {"valid_pixels": 0}

    logits = logits[mask]
    gt_flat = gt_flat[mask]

    # Raw stats
    neg_ratio = float((logits < 0).float().mean().item())
    max_vals, max_idx = torch.max(logits, dim=1)
    all_nonpos_ratio = float((max_vals <= 0).float().mean().item())

    # Clamp stats
    logits_pos = torch.clamp(logits, min=0.0)
    max_pos, max_idx_pos = torch.max(logits_pos, dim=1)
    argmax_flip = float((max_idx != max_idx_pos).float().mean().item())

    # Mass + margin
    sum_pos = logits_pos.sum(dim=1)
    top2 = torch.topk(logits, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]

    # Entropy on normalized positive logits
    norm_pos = positive_normalize(logits_pos, dim=1, min=0.0)
    norm_pos = torch.clamp(norm_pos, min=1e-6)
    entropy = -torch.sum(norm_pos * torch.log(norm_pos), dim=1)

    stats = {
        "valid_pixels": int(mask.sum().item()),
        "neg_ratio": neg_ratio,
        "all_nonpos_ratio": all_nonpos_ratio,
        "argmax_flip_ratio": argmax_flip,
        "sum_pos_mean": float(sum_pos.mean().item()),
        "margin_mean": float(margin.mean().item()),
        "entropy_mean": float(entropy.mean().item()),
    }
    stats.update({f"sum_pos_{k}": v for k, v in _quantiles(sum_pos, [0.01, 0.5, 0.99]).items()})
    stats.update({f"margin_{k}": v for k, v in _quantiles(margin, [0.01, 0.5, 0.99]).items()})
    stats.update({f"entropy_{k}": v for k, v in _quantiles(entropy, [0.01, 0.5, 0.99]).items()})
    return stats


def _aggregate_stats(per_frame: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_frame:
        return {}
    keys = [k for k in per_frame[0].keys() if k != "valid_pixels"]
    agg = {}
    for k in keys:
        agg[k] = float(np.mean([d[k] for d in per_frame if k in d]))
    agg["frames"] = len(per_frame)
    agg["valid_pixels_total"] = int(np.sum([d.get("valid_pixels", 0) for d in per_frame]))
    return agg


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

    per_frame_stats: List[Dict[str, float]] = []
    eval_every = slam.config.get("eval_every", 1)

    for time_idx in range(len(dataset)):
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue
        if len(per_frame_stats) >= args.max_frames:
            break

        color, _, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        intrinsics = intrinsics[:3, :3]
        seman_gt = dataset.get_semantic_map(time_idx)[0].to(slam.device)

        if time_idx == 0:
            first_frame_w2c = torch.linalg.inv(pose)
            cam = setup_camera(
                color.shape[1],
                color.shape[0],
                intrinsics.cpu().numpy(),
                first_frame_w2c.detach().cpu().numpy(),
                num_channels=n_cls,
            )

        transformed_gaussians = transform_to_frame(
            slam.params,
            time_idx,
            gaussians_grad=False,
            camera_grad=False,
            rel_w2c=gt_w2c,
        )

        rendervar = transformed_params2rendervar(slam.params, transformed_gaussians)
        _, radius, _ = Renderer(raster_settings=cam)(**rendervar)
        seen = radius > 0

        sem_cam = cam
        if "seman_cls_ids" in slam.variables:
            cls_ids = slam.variables["seman_cls_ids"].to(seen.device)
            if cls_ids.shape[0] == seen.shape[0]:
                cls_ids = cls_ids[seen]
            sem_cam = set_camera_sparse(cam=cam, cls_ids=cls_ids)

        seman_rendervar = transformed_params2semrendervar_sparse(
            slam.params, transformed_gaussians, seen
        )
        rastered, _ = SEMRenderer(raster_settings=sem_cam)(**seman_rendervar)

        stats = _frame_stats(rastered, seman_gt)
        stats["frame_id"] = time_idx
        per_frame_stats.append(stats)

    agg = _aggregate_stats(per_frame_stats)
    eval_dir = os.path.join(
        slam.config["workdir"],
        f"eval_{args.out_suffix}",
    )
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "render_stats.txt")
    with open(out_path, "w") as f:
        f.write(f"stage: {args.stage}\nstep: {args.step}\nframes: {agg.get('frames', 0)}\n")
        for k in sorted(agg.keys()):
            if k in ("frames", "valid_pixels_total"):
                continue
            f.write(f"{k}: {agg[k]:.6f}\n")
        f.write(f"valid_pixels_total: {agg.get('valid_pixels_total', 0)}\n")

    print(f"[render_stats] wrote {out_path}")


if __name__ == "__main__":
    main()
