#!/usr/bin/env python3
"""
Evaluate semantic rendering with per-pixel positive normalization (debug only).
"""

import argparse
import os
import sys

import torch
from tensorboardX import SummaryWriter

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "third_parties", "splatam"))

from src.naruto.cfg_loader import load_cfg
from src.utils.general_utils import fix_random_seed, InfoPrinter
from src.slam import init_SLAM_model

from src.slam.semsplatam.modified_ver.splatam.eval_helper import (
    transform_to_frame,
    calc_topk_acc,
    calc_mAP,
    calc_miou,
    calc_f1,
)
from src.slam.semsplatam.modified_ver.splatam.splatam import (
    setup_camera,
    set_camera_sparse,
    transformed_params2rendervar,
    transformed_params2semrendervar_sparse,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from sparse_channel_rasterization import GaussianRasterizer as SEMRenderer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic eval with per-pixel normalization")
    parser.add_argument("--cfg", type=str, required=True, help="Config file")
    parser.add_argument("--result_dir", type=str, required=True, help="Result dir for params*.npz")
    parser.add_argument("--stage", type=str, default="final", help="Checkpoint stage")
    parser.add_argument("--step", type=int, default=1100, help="Checkpoint step")
    parser.add_argument("--out_suffix", type=str, default="final_pixel_norm", help="Eval dir suffix")
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

    top1_list, top3_list, top5_list = [], [], []
    miou_list, f1_list, map_list = [], [], []

    eval_every = slam.config.get("eval_every", 1)

    for time_idx in range(len(dataset)):
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue

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

        # Per-pixel positive normalization
        rastered = torch.clamp(rastered, min=0.0)
        denom = rastered.sum(dim=0, keepdim=True).clamp(min=1e-6)
        rastered = rastered / denom

        pred_logits = rastered.permute(1, 2, 0)
        pred_cls = pred_logits.argmax(-1)

        topks = calc_topk_acc(pred_logits=pred_logits, target=seman_gt.long(), topk=(1, 3, 5))
        top1_list.append(topks[0])
        top3_list.append(topks[1])
        top5_list.append(topks[2])

        miou_list.append(calc_miou(pred=pred_cls, target=seman_gt.long()))
        f1_list.append(calc_f1(pred=pred_cls, target=seman_gt.long()))
        map_list.append(calc_mAP(pred_logits=pred_logits, target=seman_gt.long()))

    eval_dir = os.path.join(slam.config["workdir"], f"eval_{args.out_suffix}")
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "summary.txt")
    with open(out_path, "w") as f:
        f.write(f"top1: {float(torch.tensor(top1_list).mean()):.4f}\n")
        f.write(f"top3: {float(torch.tensor(top3_list).mean()):.4f}\n")
        f.write(f"top5: {float(torch.tensor(top5_list).mean()):.4f}\n")
        f.write(f"mIoU: {float(torch.tensor(miou_list).mean()):.4f}\n")
        f.write(f"F1: {float(torch.tensor(f1_list).mean()):.4f}\n")
        f.write(f"mAP: {float(torch.tensor(map_list).mean()):.4f}\n")

    print(f"[eval_pixel_norm] wrote {out_path}")


if __name__ == "__main__":
    main()
