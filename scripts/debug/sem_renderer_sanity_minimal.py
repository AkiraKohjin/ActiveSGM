#!/usr/bin/env python3
"""
Minimal sanity test for sparse semantic rasterizer:
Two Gaussians at same (x,y) but different depth and different classes.
Checks whether front Gaussian dominates.
"""

import os
import sys

import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "third_parties", "splatam"))

from src.slam.semsplatam.modified_ver.splatam.splatam import setup_camera, set_camera_sparse
from sparse_channel_rasterization import GaussianRasterizer as SEMRenderer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = 64, 64
    fx = fy = 50.0
    cx = W / 2.0
    cy = H / 2.0
    intrinsics = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
    ).numpy()
    w2c = torch.eye(4, dtype=torch.float32).numpy()

    num_channels = 4
    cam = setup_camera(W, H, intrinsics, w2c, num_channels=num_channels)

    # Two Gaussians, same (x,y), different depth
    means3D = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32, device=device
    )
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    scales = torch.tensor(
        [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]],
        dtype=torch.float32,
        device=device,
    )
    opacities = torch.tensor([[1.0], [1.0]], dtype=torch.float32, device=device)

    # TOPK is fixed at 16 in the sparse rasterizer. Fill unused with -1/0.
    TOPK = 16
    colors_precomp = torch.zeros((2, TOPK), dtype=torch.float32, device=device)
    colors_precomp[:, 0] = 1.0
    cls_ids = torch.full((2, TOPK), -1, dtype=torch.int32, device=device)
    cls_ids[0, 0] = 1  # front
    cls_ids[1, 0] = 2  # back
    cam = set_camera_sparse(cam=cam, cls_ids=cls_ids)

    rendervar = {
        "means3D": means3D,
        "colors_precomp": colors_precomp,
        "rotations": rotations,
        "opacities": opacities,
        "scales": scales,
        "means2D": torch.zeros_like(means3D, device=device),
    }

    img, _ = SEMRenderer(raster_settings=cam)(**rendervar)
    img = img.detach().cpu()
    pred = img.argmax(0)
    center = pred[H // 2, W // 2].item()
    print(f"Center class (front=1, back=2): {center}")

    # Swap depths: back becomes front
    means3D_swapped = torch.tensor(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device
    )
    rendervar["means3D"] = means3D_swapped
    img2, _ = SEMRenderer(raster_settings=cam)(**rendervar)
    img2 = img2.detach().cpu()
    pred2 = img2.argmax(0)
    center2 = pred2[H // 2, W // 2].item()
    print(f"Center class after swap (front=2): {center2}")


if __name__ == "__main__":
    main()
