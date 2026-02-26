#!/usr/bin/env python3
"""
Per-class IoU/Acc for OneFormer pseudo labels vs GT.

This script does NOT modify checkpoints on disk.
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo vs GT confusion analysis.")
    parser.add_argument("--cfg", type=str, required=True, help="Config file")
    parser.add_argument("--result_dir", type=str, required=True, help="Result dir")
    parser.add_argument("--stage", type=str, default="final", help="Checkpoint stage")
    parser.add_argument("--step", type=int, default=1100, help="Checkpoint step")
    parser.add_argument("--max_frames", type=int, default=20, help="Max evaluated frames")
    parser.add_argument("--downscale", type=int, default=4, help="Downscale factor (>=1)")
    parser.add_argument("--out_suffix", type=str, default="final_pseudo_confusion", help="Eval dir suffix")
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


def main() -> None:
    args = _parse_args()
    info_printer = InfoPrinter("ActiveSem")
    main_cfg = load_cfg(args)
    fix_random_seed(main_cfg.general.seed)

    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    _ = SummaryWriter(f"{log_savedir}")

    slam = init_SLAM_model(main_cfg, info_printer, None)
    # Load params to keep config consistent with standard eval path
    slam.load_params_by_step(step=args.step, stage=args.stage)

    dataset = slam.dataset_eval
    n_cls = slam.n_cls
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)

    semantic_dir = getattr(main_cfg.slam, "semantic_dir", "")
    class_names = _load_class_names(semantic_dir, n_cls)

    eval_every = slam.config.get("eval_every", 1)
    frame_count = 0

    for time_idx in range(len(dataset)):
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue
        if frame_count >= args.max_frames:
            break

        color, _, _, _ = dataset[time_idx]
        gt = dataset.get_semantic_map(time_idx)[0]

        # OneFormer pseudo labels
        seg_img = color.clone().to(slam.semantic_device)
        pseudo_id, _ = slam.semantic_annotation(seg_img)
        pseudo_id = pseudo_id.to(gt.device)

        if args.downscale > 1:
            gt = gt[:: args.downscale, :: args.downscale]
            pseudo_id = pseudo_id[:: args.downscale, :: args.downscale]

        _update_confusion(conf, pseudo_id.detach().cpu().numpy(), gt.detach().cpu().numpy())
        frame_count += 1

    eval_dir = slam.eval_dir + "_" + args.out_suffix
    os.makedirs(eval_dir, exist_ok=True)
    np.save(os.path.join(eval_dir, "confusion.npy"), conf)

    diag = np.diag(conf)
    gt_sum = conf.sum(axis=1)
    pred_sum = conf.sum(axis=0)
    iou = diag / (gt_sum + pred_sum - diag + 1e-9)
    acc = diag / (gt_sum + 1e-9)

    # mIoU over classes present in GT
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

    print(f"[sem_pseudo_confusion] wrote {eval_dir}")


if __name__ == "__main__":
    main()
