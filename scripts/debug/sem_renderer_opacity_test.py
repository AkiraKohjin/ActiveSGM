#!/usr/bin/env python3
"""
Test whether the sparse semantic rasterizer respects per-Gaussian opacities.
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

from src.slam.semsplatam.modified_ver.splatam.eval_helper import transform_to_frame
from src.slam.semsplatam.modified_ver.splatam.splatam import (
    setup_camera,
    set_camera_sparse,
    transformed_params2rendervar,
    transformed_params2semrendervar_sparse,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from sparse_channel_rasterization import GaussianRasterizer as SEMRenderer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic renderer opacity test")
    parser.add_argument("--cfg", type=str, required=True, help="Config file")
    parser.add_argument("--result_dir", type=str, required=True, help="Result dir for params*.npz")
    parser.add_argument("--stage", type=str, default="final", help="Checkpoint stage")
    parser.add_argument("--step", type=int, default=1100, help="Checkpoint step")
    parser.add_argument("--frame_idx", type=int, default=50, help="Frame index to test")
    parser.add_argument("--out_suffix", type=str, default="final_opacity_test", help="Eval dir suffix")
    return parser.parse_args()


def _stat(tag: str, t: torch.Tensor) -> str:
    finite = torch.isfinite(t)
    bad = int((~finite).sum().item())
    if finite.any():
        vals = t[finite]
        vmin = float(vals.min().item())
        vmax = float(vals.max().item())
        vmean = float(vals.mean().item())
    else:
        vmin = vmax = vmean = float("nan")
    return f"{tag}: bad={bad} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}"


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
    frame_idx = min(args.frame_idx, len(dataset) - 1)

    color, _, intrinsics, pose = dataset[frame_idx]
    gt_w2c = torch.linalg.inv(pose)
    intrinsics = intrinsics[:3, :3]

    first_frame_w2c = torch.linalg.inv(dataset[0][3])
    cam = setup_camera(
        color.shape[1],
        color.shape[0],
        intrinsics.cpu().numpy(),
        first_frame_w2c.detach().cpu().numpy(),
        num_channels=n_cls,
    )

    transformed_gaussians = transform_to_frame(
        slam.params,
        frame_idx,
        gaussians_grad=False,
        camera_grad=False,
        rel_w2c=gt_w2c,
    )

    rendervar = transformed_params2rendervar(slam.params, transformed_gaussians)
    _, radius, _ = Renderer(raster_settings=cam)(**rendervar)
    seen = radius > 0

    sem_cam = cam
    cls_ids = slam.variables["seman_cls_ids"].to(seen.device)
    if cls_ids.shape[0] == seen.shape[0]:
        cls_ids = cls_ids[seen]
    sem_cam = set_camera_sparse(cam=cam, cls_ids=cls_ids)

    seman_rendervar = transformed_params2semrendervar_sparse(
        slam.params, transformed_gaussians, seen
    )

    # Original
    sem_orig, _ = SEMRenderer(raster_settings=sem_cam)(**seman_rendervar)
    # Zero opacities
    rend_zero = dict(seman_rendervar)
    rend_zero["opacities"] = torch.zeros_like(seman_rendervar["opacities"])
    sem_zero, _ = SEMRenderer(raster_settings=sem_cam)(**rend_zero)
    # One opacities
    rend_one = dict(seman_rendervar)
    rend_one["opacities"] = torch.ones_like(seman_rendervar["opacities"])
    sem_one, _ = SEMRenderer(raster_settings=sem_cam)(**rend_one)

    eval_dir = os.path.join(slam.config["workdir"], f"eval_{args.out_suffix}")
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "opacity_test.txt")
    with open(out_path, "w") as f:
        f.write(_stat("orig", sem_orig) + "\n")
        f.write(_stat("opac_zero", sem_zero) + "\n")
        f.write(_stat("opac_one", sem_one) + "\n")
    print(f"[opacity_test] wrote {out_path}")


if __name__ == "__main__":
    main()
