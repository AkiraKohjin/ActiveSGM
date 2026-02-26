#!/usr/bin/env python3
"""
Inspect semantic parameter statistics (per-gaussian).

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
from src.slam.semsplatam.modified_ver.semantic.oneformer import positive_normalize


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic parameter statistics.")
    parser.add_argument("--cfg", type=str, required=True, help="Config file")
    parser.add_argument("--result_dir", type=str, required=True, help="Result dir for params*.npz")
    parser.add_argument("--stage", type=str, default="final", help="Checkpoint stage")
    parser.add_argument("--step", type=int, default=1100, help="Checkpoint step")
    parser.add_argument("--out_suffix", type=str, default="final_param_stats", help="Eval dir suffix")
    return parser.parse_args()


def _quantiles(x: torch.Tensor, qs: List[float]) -> Dict[str, float]:
    if x.numel() == 0:
        return {f"q{int(q*100)}": 0.0 for q in qs}
    vals = torch.quantile(x, torch.tensor(qs, device=x.device))
    return {f"q{int(q*100)}": float(v.item()) for q, v in zip(qs, vals)}


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

    sem = slam.params.get("semantic_logits", None)
    if sem is None:
        raise RuntimeError("semantic_logits not found in params")

    # Per-gaussian stats (TOPK channels)
    sem = sem.detach()
    neg_ratio = float((sem < 0).float().mean().item())

    sem_pos = torch.clamp(sem, min=0.0)
    sum_pos = sem_pos.sum(dim=1)
    top2 = torch.topk(sem, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]

    norm_pos = positive_normalize(sem_pos, dim=1, min=0.0)
    norm_pos = torch.clamp(norm_pos, min=1e-6)
    entropy = -torch.sum(norm_pos * torch.log(norm_pos), dim=1)
    max_prob = torch.max(norm_pos, dim=1).values

    opac = None
    if "logit_opacities" in slam.params:
        opac = torch.sigmoid(slam.params["logit_opacities"].detach()).squeeze(-1)

    stats = {
        "num_gaussians": int(sem.shape[0]),
        "topk": int(sem.shape[1]),
        "neg_ratio": neg_ratio,
        "sum_pos_mean": float(sum_pos.mean().item()),
        "margin_mean": float(margin.mean().item()),
        "entropy_mean": float(entropy.mean().item()),
        "max_prob_mean": float(max_prob.mean().item()),
    }
    stats.update({f"sum_pos_{k}": v for k, v in _quantiles(sum_pos, [0.01, 0.5, 0.99]).items()})
    stats.update({f"margin_{k}": v for k, v in _quantiles(margin, [0.01, 0.5, 0.99]).items()})
    stats.update({f"entropy_{k}": v for k, v in _quantiles(entropy, [0.01, 0.5, 0.99]).items()})
    stats.update({f"max_prob_{k}": v for k, v in _quantiles(max_prob, [0.01, 0.5, 0.99]).items()})

    if opac is not None:
        stats["opacity_mean"] = float(opac.mean().item())
        stats.update({f"opacity_{k}": v for k, v in _quantiles(opac, [0.01, 0.5, 0.99]).items()})

    eval_dir = os.path.join(
        slam.config["workdir"],
        f"eval_{args.out_suffix}",
    )
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "param_stats.txt")
    with open(out_path, "w") as f:
        f.write(f"stage: {args.stage}\nstep: {args.step}\n")
        for k in sorted(stats.keys()):
            f.write(f"{k}: {stats[k]}\n")

    print(f"[param_stats] wrote {out_path}")


if __name__ == "__main__":
    main()
