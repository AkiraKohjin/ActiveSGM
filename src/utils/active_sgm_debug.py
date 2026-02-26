import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.utils.general_utils import create_class_colormap


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val in ("1", "true", "True")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


class SemanticDebugHelper:
    """Centralized debug helpers for semantic logging and visualization."""

    def __init__(
        self,
        result_dir: str,
        n_cls: int,
        class_id_to_name: Dict[int, str],
        device: torch.device,
        semantic_device: torch.device,
    ) -> None:
        self.n_cls = n_cls
        self.class_id_to_name = class_id_to_name
        self.device = device
        self.semantic_device = semantic_device

        # Debug semantic saving (off by default)
        self.debug_semantic = env_flag("ACTIVE_SGM_DEBUG_SEM", False)
        self.debug_semantic_every = env_int("ACTIVE_SGM_DEBUG_SEM_EVERY", 1)
        self.debug_semantic_max = env_int("ACTIVE_SGM_DEBUG_SEM_MAX", -1)
        self.debug_semantic_dir = os.path.join(result_dir, "debug_semantic")

        # Debug semantic stats (off by default)
        self.debug_seman_stats = env_flag("ACTIVE_SGM_DEBUG_SEMAN_STATS", False)
        self.debug_seman_stats_every = env_int("ACTIVE_SGM_DEBUG_SEMAN_STATS_EVERY", 1)
        self.debug_seman_stats_max = env_int("ACTIVE_SGM_DEBUG_SEMAN_STATS_MAX", -1)

        # GPU peer-access info
        self.debug_seman_p2p = env_flag("ACTIVE_SGM_DEBUG_SEMAN_P2P", False)

        # Internal counters
        self._oneformer_debug_count = 0
        self._seman_transfer_count = 0
        self._render_sem_debug_count = 0
        self._param_sem_debug_count = 0

        if self.debug_seman_stats or self.debug_seman_p2p:
            self._log_device_info()

    def _log_device_info(self) -> None:
        try:
            dev_cnt = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print(f"[ActiveSGM][seman-stats] cuda_available={torch.cuda.is_available()} device_count={dev_cnt}")
            print(f"[ActiveSGM][seman-stats] primary_device={self.device} semantic_device={self.semantic_device}")
            if torch.cuda.is_available():
                for d in range(dev_cnt):
                    props = torch.cuda.get_device_properties(d)
                    print(f"[ActiveSGM][seman-stats] dev{d} name={props.name} total_mem={props.total_memory}")
            if torch.cuda.is_available() and dev_cnt >= 2:
                for src in range(dev_cnt):
                    for dst in range(dev_cnt):
                        if src == dst:
                            continue
                        can = torch.cuda.can_device_access_peer(src, dst)
                        if self.debug_seman_p2p:
                            print(f"[ActiveSGM][seman-stats] can_access_peer {src}->{dst}: {can}")
            if self.debug_seman_p2p and torch.cuda.is_available():
                try:
                    sem_dev = torch.device(self.semantic_device)
                    t = torch.rand((4, 4), device=sem_dev)
                    t2 = t.to(self.device)
                    max_diff = (t2 - t.to(self.device)).abs().max().item()
                    print(
                        "[ActiveSGM][seman-stats] "
                        f"p2p_copy_test max_diff={max_diff:.6f} "
                        f"sum_src={t.sum().item():.6f} sum_dst={t2.sum().item():.6f}"
                    )
                except Exception as exc:
                    print(f"[ActiveSGM][seman-stats] p2p_copy_test failed: {exc}")
        except Exception as exc:
            print(f"[ActiveSGM][seman-stats] device info failed: {exc}")

    def log_seman_stats(self, seman: torch.Tensor, tag: str, time_idx: int) -> None:
        if not self.debug_seman_stats:
            return
        if self.debug_seman_stats_every <= 0:
            return
        if time_idx % self.debug_seman_stats_every != 0:
            return
        if self.debug_seman_stats_max >= 0 and time_idx > self.debug_seman_stats_max:
            return
        if not isinstance(seman, torch.Tensor):
            print(f"[ActiveSGM][seman-stats] {tag} t={time_idx}: not a tensor ({type(seman)})")
            return
        with torch.no_grad():
            try:
                shape = tuple(seman.shape)
                dtype = seman.dtype
                device = seman.device
                stride = seman.stride() if hasattr(seman, "stride") else None
                is_contig = seman.is_contiguous() if hasattr(seman, "is_contiguous") else None
                seman_f = seman.float()
                min_v = seman_f.min().item()
                max_v = seman_f.max().item()
                mean_v = seman_f.mean().item()
                # Per-pixel sum over channels if shape is HxWxC or CxHxW
                sum_v = None
                if seman_f.dim() == 3:
                    if seman_f.shape[0] <= 512 and seman_f.shape[0] <= seman_f.shape[-1]:
                        # Heuristic: assume CxHxW
                        sum_v = seman_f.sum(dim=0).mean().item()
                    else:
                        # Assume HxWxC
                        sum_v = seman_f.sum(dim=-1).mean().item()
                print(
                    "[ActiveSGM][seman-stats] "
                    f"{tag} t={time_idx} shape={shape} dtype={dtype} device={device} "
                    f"min={min_v:.4f} max={max_v:.4f} mean={mean_v:.4f}"
                    + (f" mean_sum={sum_v:.4f}" if sum_v is not None else "")
                    + (f" contiguous={is_contig} stride={stride}" if stride is not None else "")
                )
            except Exception as exc:
                print(f"[ActiveSGM][seman-stats] {tag} t={time_idx}: failed ({exc})")

    def log_isfinite(self, seman: torch.Tensor, tag: str, time_idx: int) -> None:
        if not self.debug_seman_stats:
            return
        try:
            print(
                f"[ActiveSGM][seman-stats] {tag} isfinite={bool(torch.isfinite(seman).all())} t={time_idx}"
            )
        except Exception as exc:
            print(f"[ActiveSGM][seman-stats] {tag} isfinite check failed t={time_idx}: {exc}")

    def log_contig_check(self, seman: torch.Tensor, time_idx: int) -> None:
        if not self.debug_seman_stats:
            return
        try:
            seman_contig = seman.contiguous()
            self.log_seman_stats(seman_contig, "seman_chw_contig", time_idx)
            if not torch.isfinite(seman).all():
                print(f"[ActiveSGM][seman-stats] seman_chw has non-finite values at t={time_idx}")
            if not torch.isfinite(seman_contig).all():
                print(f"[ActiveSGM][seman-stats] seman_chw_contig has non-finite values at t={time_idx}")
        except Exception as exc:
            print(f"[ActiveSGM][seman-stats] contig check failed at t={time_idx}: {exc}")

    def log_seman_transfer(self, src: torch.Tensor, dst: torch.Tensor, tag: str, time_idx: int) -> None:
        if not env_flag("ACTIVE_SGM_DEBUG_SEMAN_TRANSFER", False):
            return
        max_logs = env_int("ACTIVE_SGM_DEBUG_SEMAN_TRANSFER_MAX", 5)
        if self._seman_transfer_count >= max_logs:
            return
        self._seman_transfer_count += 1
        try:
            def _nf_stats(t: torch.Tensor) -> Tuple[int, float, float, float]:
                finite = torch.isfinite(t)
                bad = int((~finite).sum().item())
                if finite.any():
                    vals = t[finite]
                    vmin = float(vals.min().item())
                    vmax = float(vals.max().item())
                    vmean = float(vals.mean().item())
                else:
                    vmin = vmax = vmean = float("nan")
                return bad, vmin, vmax, vmean

            src_bad, src_min, src_max, src_mean = _nf_stats(src)
            dst_bad, dst_min, dst_max, dst_mean = _nf_stats(dst)
            print(
                "[ActiveSGM][seman-transfer] "
                f"{tag} t={time_idx} "
                f"src_dev={src.device} dst_dev={dst.device} "
                f"src_bad={src_bad} dst_bad={dst_bad} "
                f"src_min={src_min:.6f} src_max={src_max:.6f} src_mean={src_mean:.6f} "
                f"dst_min={dst_min:.6f} dst_max={dst_max:.6f} dst_mean={dst_mean:.6f}"
            )
        except Exception as exc:
            print(f"[ActiveSGM][seman-transfer] {tag} t={time_idx} failed ({exc})")

    def log_oneformer_stats(
        self,
        logits: torch.Tensor,
        info_printer,
        step: int,
        class_name: str,
    ) -> None:
        if not env_flag("ACTIVE_SGM_DEBUG_ONEFORMER", False):
            return
        max_logs = env_int("ACTIVE_SGM_DEBUG_ONEFORMER_MAX", 20)
        mode = env_str("ACTIVE_SGM_DEBUG_ONEFORMER_MODE", "nonfinite").lower()
        if self._oneformer_debug_count >= max_logs:
            return
        if not isinstance(logits, torch.Tensor):
            return
        finite = torch.isfinite(logits)
        any_bad = (~finite).any().item()
        if mode == "nonfinite" and not any_bad:
            return
        self._oneformer_debug_count += 1
        bad = int((~finite).sum().item())
        if finite.any():
            vals = logits[finite]
            vmin = float(vals.min().item())
            vmax = float(vals.max().item())
            vmean = float(vals.mean().item())
        else:
            vmin = vmax = vmean = float("nan")
        info_printer(
            f"[OneFormerDebug] bad={bad} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}",
            step,
            class_name,
        )

    def save_semantic_debug(
        self,
        time_idx: int,
        color: torch.Tensor,
        class_ids: torch.Tensor,
        seman: torch.Tensor,
    ) -> None:
        if not self.debug_semantic:
            return
        if self.debug_semantic_every <= 0:
            return
        if time_idx % self.debug_semantic_every != 0:
            return
        if self.debug_semantic_max >= 0 and time_idx > self.debug_semantic_max:
            return

        os.makedirs(self.debug_semantic_dir, exist_ok=True)
        self._save_semantic_legend()
        frame_dir = os.path.join(self.debug_semantic_dir, f"{time_idx:04d}")
        os.makedirs(frame_dir, exist_ok=True)

        # RGB
        rgb = color.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
        Image.fromarray(rgb).save(os.path.join(frame_dir, "rgb.png"))

        # Class IDs
        cls = class_ids.detach().cpu().numpy().astype(np.int32)
        cls = np.clip(cls, 0, self.n_cls - 1)
        Image.fromarray(cls.astype(np.uint8)).save(os.path.join(frame_dir, "semantic_ids.png"))

        # Colorized semantic + overlay
        colormap = create_class_colormap(self.n_cls)
        sem_color = colormap[cls]
        Image.fromarray(sem_color).save(os.path.join(frame_dir, "semantic_color.png"))
        overlay = (0.5 * rgb + 0.5 * sem_color).astype(np.uint8)
        Image.fromarray(overlay).save(os.path.join(frame_dir, "overlay.png"))

        # Confidence map (max prob)
        if isinstance(seman, torch.Tensor):
            conf = seman.detach().max(dim=-1).values
            conf = conf.clamp(0, 1).mul(255).byte().cpu().numpy()
            Image.fromarray(conf).save(os.path.join(frame_dir, "semantic_conf.png"))

        # Histogram
        hist = np.bincount(cls.reshape(-1), minlength=self.n_cls)
        with open(os.path.join(frame_dir, "semantic_hist.csv"), "w") as f:
            f.write("class_id,count\n")
            for cid, cnt in enumerate(hist.tolist()):
                f.write(f"{cid},{cnt}\n")

    def _save_semantic_legend(self) -> None:
        legend_path = os.path.join(self.debug_semantic_dir, "semantic_color_legend.png")
        csv_path = os.path.join(self.debug_semantic_dir, "semantic_color_legend.csv")
        if os.path.exists(legend_path) and os.path.exists(csv_path):
            return

        os.makedirs(self.debug_semantic_dir, exist_ok=True)
        colormap = create_class_colormap(self.n_cls)

        cols = 4
        cell_h = 20
        cell_w = 320
        swatch = 14
        pad = 3
        rows = int(math.ceil(self.n_cls / cols))
        width = cols * cell_w
        height = rows * cell_h

        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        for cid in range(self.n_cls):
            row = cid // cols
            col = cid % cols
            x0 = col * cell_w
            y0 = row * cell_h
            color = tuple(int(x) for x in colormap[cid].tolist())
            draw.rectangle([x0 + pad, y0 + pad, x0 + pad + swatch, y0 + pad + swatch], fill=color)
            name = self.class_id_to_name.get(cid, "unknown")
            label = f"{cid}: {name}"
            draw.text((x0 + pad + swatch + 6, y0 + pad), label, fill=(0, 0, 0), font=font)

        img.save(legend_path)

        with open(csv_path, "w") as f:
            f.write("class_id,name,r,g,b\n")
            for cid in range(self.n_cls):
                name = self.class_id_to_name.get(cid, "unknown")
                r, g, b = [int(x) for x in colormap[cid].tolist()]
                f.write(f"{cid},{name},{r},{g},{b}\n")

    def maybe_sync_seman(self, seman: torch.Tensor, tag: str) -> None:
        if not env_flag("ACTIVE_SGM_DEBUG_SEMAN_SYNC", False):
            return
        try:
            print(
                f"[ActiveSGM][seman-debug] seman ({tag}): "
                f"shape={tuple(seman.shape)} dtype={seman.dtype} device={seman.device}"
            )
            if seman.is_cuda:
                torch.cuda.synchronize()
        except Exception as exc:
            print(f"[ActiveSGM][seman-debug] {tag} sync failed: {exc}")

    def maybe_cpu_check(self, seman: torch.Tensor) -> None:
        if not env_flag("ACTIVE_SGM_DEBUG_SEMAN_CPU_CHECK", False):
            return
        try:
            seman_cpu = seman.detach().float().cpu()
            if not torch.isfinite(seman_cpu).all():
                print("[ActiveSGM][seman-debug] seman contains NaN/Inf on CPU copy")
        except Exception as exc:
            print(f"[ActiveSGM][seman-debug] seman CPU check failed: {exc}")

    def log_render_sem_inputs(
        self,
        seen: torch.Tensor,
        seman_rendervar: Dict[str, torch.Tensor],
        info_printer,
        step: int,
        class_name: str,
    ) -> None:
        if not env_flag("ACTIVE_SGM_DEBUG_RENDER_SEM", False):
            return
        max_logs = env_int("ACTIVE_SGM_DEBUG_RENDER_SEM_MAX", 20)
        mode = env_str("ACTIVE_SGM_DEBUG_RENDER_SEM_MODE", "nonfinite").lower()
        if self._render_sem_debug_count >= max_logs:
            return

        def _tensor_stats(t: torch.Tensor) -> Tuple[int, float, float, float]:
            finite = torch.isfinite(t)
            bad = int((~finite).sum().item())
            if finite.any():
                vals = t[finite]
                return bad, float(vals.min().item()), float(vals.max().item()), float(vals.mean().item())
            return bad, float("nan"), float("nan"), float("nan")

        any_bad = False
        stats = {}
        for k, v in {
            "means3D": seman_rendervar["means3D"],
            "colors_precomp": seman_rendervar["colors_precomp"],
            "rotations": seman_rendervar["rotations"],
            "opacities": seman_rendervar["opacities"],
            "scales": seman_rendervar["scales"],
        }.items():
            bad, vmin, vmax, vmean = _tensor_stats(v)
            stats[k] = (bad, vmin, vmax, vmean)
            if bad > 0:
                any_bad = True

        if mode == "nonfinite" and not any_bad:
            return
        self._render_sem_debug_count += 1

        msg = f"[RenderSemDebug] pre seen={int(seen.sum().item())}"
        for k, (bad, vmin, vmax, vmean) in stats.items():
            msg += f" {k}:bad={bad} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f};"
        info_printer(msg, step, class_name)

    def log_render_sem_logits(
        self,
        logits: torch.Tensor,
        info_printer,
        step: int,
        class_name: str,
    ) -> None:
        if not env_flag("ACTIVE_SGM_DEBUG_RENDER_SEM", False):
            return
        max_logs = env_int("ACTIVE_SGM_DEBUG_RENDER_SEM_MAX", 20)
        mode = env_str("ACTIVE_SGM_DEBUG_RENDER_SEM_MODE", "nonfinite").lower()
        if self._render_sem_debug_count >= max_logs:
            return
        finite = torch.isfinite(logits)
        bad = int((~finite).sum().item())
        if mode == "nonfinite" and bad == 0:
            return
        self._render_sem_debug_count += 1
        if finite.any():
            vals = logits[finite]
            vmin = float(vals.min().item())
            vmax = float(vals.max().item())
            vmean = float(vals.mean().item())
        else:
            vmin = vmax = vmean = float("nan")
        info_printer(
            f"[RenderSemDebug] post_logits bad={bad} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}",
            step,
            class_name,
        )

    def param_sem_debug_enabled(self) -> bool:
        return env_flag("ACTIVE_SGM_DEBUG_PARAM_SEM", False)

    def log_param_sem(
        self,
        params: Dict[str, torch.Tensor],
        tag: str,
        iter_idx: int,
        info_printer,
        step: int,
        class_name: str,
    ) -> None:
        if not self.param_sem_debug_enabled():
            return
        max_logs = env_int("ACTIVE_SGM_DEBUG_PARAM_SEM_MAX", 20)
        if self._param_sem_debug_count >= max_logs:
            return
        try:
            sem_logits = params.get("semantic_logits", None)
            if sem_logits is None:
                return
            finite = torch.isfinite(sem_logits)
            bad = int((~finite).sum().item())
            if finite.any():
                vals = sem_logits[finite]
                vmin = float(vals.min().item())
                vmax = float(vals.max().item())
                vmean = float(vals.mean().item())
            else:
                vmin = vmax = vmean = float("nan")
            grad = sem_logits.grad if hasattr(sem_logits, "grad") else None
            g_bad = None
            g_min = g_max = g_mean = None
            if grad is not None:
                g_fin = torch.isfinite(grad)
                g_bad = int((~g_fin).sum().item())
                if g_fin.any():
                    g_vals = grad[g_fin]
                    g_min = float(g_vals.min().item())
                    g_max = float(g_vals.max().item())
                    g_mean = float(g_vals.mean().item())
                else:
                    g_min = g_max = g_mean = float("nan")
            msg = f"[ParamSemDebug] {tag} iter={iter_idx} bad={bad} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}"
            if g_bad is not None:
                msg += f" grad_bad={g_bad} grad_min={g_min:.6f} grad_max={g_max:.6f} grad_mean={g_mean:.6f}"
            info_printer(msg, step, class_name)
            self._param_sem_debug_count += 1
        except Exception as exc:
            info_printer(f"[ParamSemDebug] {tag} failed ({exc})", step, class_name)

    def log_param_sem_nonfinite(
        self,
        params: Dict[str, torch.Tensor],
        iter_idx: int,
        info_printer,
        step: int,
        class_name: str,
    ) -> None:
        if not self.param_sem_debug_enabled():
            return
        try:
            if not torch.isfinite(params["semantic_logits"]).all():
                self.log_param_sem(params, f"nonfinite_after_step iter={iter_idx}", iter_idx, info_printer, step, class_name)
        except Exception as exc:
            info_printer(f"[ParamSemDebug] nonfinite check failed ({exc})", step, class_name)


_COUNTERS: Dict[str, int] = {}


def _bump_counter(key: str) -> int:
    _COUNTERS[key] = _COUNTERS.get(key, 0) + 1
    return _COUNTERS[key]


def _counter_val(key: str) -> int:
    return _COUNTERS.get(key, 0)


def log_pointcloud_seman_stats(seman: torch.Tensor) -> None:
    if not env_flag("ACTIVE_SGM_DEBUG_SEMAN_STATS", False):
        return
    try:
        if isinstance(seman, torch.Tensor):
            seman_f = seman.float()
            stride = seman_f.stride() if hasattr(seman_f, "stride") else None
            is_contig = seman_f.is_contiguous() if hasattr(seman_f, "is_contiguous") else None
            shape = tuple(seman_f.shape)
            finite = torch.isfinite(seman_f)
            bad = int((~finite).sum().item())
            if finite.any():
                vals = seman_f[finite]
                min_v = vals.min().item()
                max_v = vals.max().item()
                mean_v = vals.mean().item()
            else:
                min_v = max_v = mean_v = float("nan")
            sum_v = None
            if seman_f.dim() == 3:
                if seman_f.shape[0] <= 512 and seman_f.shape[0] <= seman_f.shape[-1]:
                    sum_v = seman_f.sum(dim=0).mean().item()
                else:
                    sum_v = seman_f.sum(dim=-1).mean().item()
            print(
                "[ActiveSGM][seman-stats] "
                f"get_pointcloud_with_seman in shape={shape} "
                f"bad={bad} min={min_v:.4f} max={max_v:.4f} mean={mean_v:.4f}"
                + (f" mean_sum={sum_v:.4f}" if sum_v is not None else "")
                + (f" contiguous={is_contig} stride={stride}" if stride is not None else "")
            )
    except Exception as exc:
        print(f"[ActiveSGM][seman-stats] get_pointcloud_with_seman failed ({exc})")


def log_init_sem_dense(sem_dense: torch.Tensor, tag: str) -> None:
    if not env_flag("ACTIVE_SGM_DEBUG_INIT_SEM", False):
        return
    try:
        finite = torch.isfinite(sem_dense)
        bad = int((~finite).sum().item())
        if finite.any():
            vals = sem_dense[finite]
            vmin = float(vals.min().item())
            vmax = float(vals.max().item())
            vmean = float(vals.mean().item())
        else:
            vmin = vmax = vmean = float("nan")
        print(f"[ActiveSGM][{tag}] dense bad={bad} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}")
    except Exception as exc:
        print(f"[ActiveSGM][{tag}] dense stat failed ({exc})")


def log_init_sem_sparse(sparse_vals: torch.Tensor, tag: str) -> None:
    if not env_flag("ACTIVE_SGM_DEBUG_INIT_SEM", False):
        return
    try:
        finite = torch.isfinite(sparse_vals)
        bad = int((~finite).sum().item())
        if finite.any():
            vals = sparse_vals[finite]
            vmin = float(vals.min().item())
            vmax = float(vals.max().item())
            vmean = float(vals.mean().item())
        else:
            vmin = vmax = vmean = float("nan")
        print(f"[ActiveSGM][{tag}] sparse bad={bad} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}")
    except Exception as exc:
        print(f"[ActiveSGM][{tag}] sparse stat failed ({exc})")


def log_sem_bg(cam) -> None:
    if not env_flag("ACTIVE_SGM_DEBUG_SEM_BG", False):
        return
    max_logs = env_int("ACTIVE_SGM_DEBUG_SEM_BG_MAX", 5)
    if _counter_val("sem_bg") >= max_logs:
        return
    _bump_counter("sem_bg")
    try:
        bg = cam.bg
        num_ch = int(cam.num_channels)
        bg_numel = int(bg.numel()) if hasattr(bg, "numel") else -1
        bg_shape = tuple(bg.shape) if hasattr(bg, "shape") else None
        finite = torch.isfinite(bg) if isinstance(bg, torch.Tensor) else None
        bad = int((~finite).sum().item()) if finite is not None else -1
        if finite is not None and finite.any():
            vals = bg[finite]
            vmin = float(vals.min().item())
            vmax = float(vals.max().item())
            vmean = float(vals.mean().item())
        else:
            vmin = vmax = vmean = float("nan")
        warn = ""
        if bg_numel >= 0 and bg_numel < num_ch:
            warn = " WARN:bg_numel<num_channels"
        print(
            "[SeManBGDebug] "
            f"bg_shape={bg_shape} bg_numel={bg_numel} num_channels={num_ch} "
            f"bad={bad} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}{warn}"
        )
    except Exception as exc:
        print(f"[SeManBGDebug] failed ({exc})")


def log_entropy_input(prob_dist: torch.Tensor) -> None:
    if not env_flag("ACTIVE_SGM_DEBUG_ENTROPY_FUNC", False):
        return
    max_logs = env_int("ACTIVE_SGM_DEBUG_ENTROPY_FUNC_MAX", 20)
    if _counter_val("entropy_func") >= max_logs:
        return
    nonfinite = ~torch.isfinite(prob_dist)
    if not nonfinite.any():
        return
    _bump_counter("entropy_func")
    bad = int(nonfinite.sum().item())
    total = int(prob_dist.numel())
    try:
        min_val = float(prob_dist[~nonfinite].min().item()) if bad < total else float("nan")
        max_val = float(prob_dist[~nonfinite].max().item()) if bad < total else float("nan")
        mean_val = float(prob_dist[~nonfinite].mean().item()) if bad < total else float("nan")
    except Exception:
        min_val = max_val = mean_val = float("nan")
    print(
        "[EntropyDebugFunc] "
        f"nonfinite in prob_dist: bad={bad}/{total} "
        f"min={min_val:.6f} max={max_val:.6f} mean={mean_val:.6f}"
    )


def log_seman_rendervar(seman_rendervar: Dict[str, torch.Tensor], cls_ids: Optional[torch.Tensor]) -> None:
    if not env_flag("ACTIVE_SGM_DEBUG_SEMAN_RENDERVAR", False):
        return
    max_logs = env_int("ACTIVE_SGM_DEBUG_SEMAN_RENDERVAR_MAX", 10)
    mode = env_str("ACTIVE_SGM_DEBUG_SEMAN_RENDERVAR_MODE", "nonfinite").lower()
    if _counter_val("seman_rendervar") >= max_logs:
        return

    def _stats(t: torch.Tensor) -> Tuple[int, float, float, float]:
        finite = torch.isfinite(t)
        bad = int((~finite).sum().item())
        if finite.any():
            vals = t[finite]
            vmin = float(vals.min().item())
            vmax = float(vals.max().item())
            vmean = float(vals.mean().item())
        else:
            vmin = vmax = vmean = float("nan")
        return bad, vmin, vmax, vmean

    fields = {
        "means3D": seman_rendervar["means3D"],
        "colors_precomp": seman_rendervar["colors_precomp"],
        "rotations": seman_rendervar["rotations"],
        "opacities": seman_rendervar["opacities"],
        "scales": seman_rendervar["scales"],
    }
    any_bad = False
    parts = []
    for k, v in fields.items():
        bad, vmin, vmax, vmean = _stats(v)
        parts.append(f"{k}:bad={bad} min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}")
        if bad > 0:
            any_bad = True
    if mode == "nonfinite" and not any_bad:
        return
    _bump_counter("seman_rendervar")
    if cls_ids is not None:
        cls_min = int(cls_ids.min().item())
        cls_max = int(cls_ids.max().item())
        cls_shape = tuple(cls_ids.shape)
        cls_msg = f" cls_ids shape={cls_shape} min={cls_min} max={cls_max}"
    else:
        cls_msg = " cls_ids missing"
    print("[SeManRenderVarDebug] " + " ".join(parts) + cls_msg)


def log_seman_repeat(
    seman_im: torch.Tensor,
    seman_rendervar: Dict[str, torch.Tensor],
    sparse_cam,
    sem_renderer,
) -> None:
    if not env_flag("ACTIVE_SGM_DEBUG_SEM_REPEAT", False):
        return
    max_logs = env_int("ACTIVE_SGM_DEBUG_SEM_REPEAT_MAX", 1)
    if _counter_val("seman_repeat") >= max_logs:
        return
    _bump_counter("seman_repeat")
    try:
        seman_im2, _, = sem_renderer(raster_settings=sparse_cam)(**seman_rendervar)
        nan1 = torch.isnan(seman_im)
        nan2 = torch.isnan(seman_im2)
        nan_any1 = nan1.any(dim=0)
        nan_any2 = nan2.any(dim=0)
        nan_cnt1 = int(nan_any1.sum().item())
        nan_cnt2 = int(nan_any2.sum().item())
        nan_xor = int((nan_any1 ^ nan_any2).sum().item())
        finite_mask = ~(nan1 | nan2)
        if finite_mask.any():
            diffs = (seman_im - seman_im2).abs()[finite_mask]
            max_diff = float(diffs.max().item())
            mean_diff = float(diffs.mean().item())
        else:
            max_diff = float("nan")
            mean_diff = float("nan")
        print(
            "[SeManRepeatDebug] "
            f"nan1={nan_cnt1} nan2={nan_cnt2} nan_xor={nan_xor} "
            f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}"
        )
    except Exception as exc:
        print(f"[SeManRepeatDebug] failed ({exc})")


def log_seman_sanity(
    seman_rendervar: Dict[str, torch.Tensor],
    sparse_cam,
    sem_renderer,
    cam_zero,
) -> None:
    if not env_flag("ACTIVE_SGM_DEBUG_SEM_SANITY", False):
        return
    if cam_zero is None:
        return
    max_logs = env_int("ACTIVE_SGM_DEBUG_SEM_SANITY_MAX", 1)
    if _counter_val("seman_sanity") >= max_logs:
        return
    _bump_counter("seman_sanity")
    try:
        def _nan_stats(tag: str, t: torch.Tensor) -> None:
            nan_any = torch.isnan(t).any(dim=0)
            nan_cnt = int(nan_any.sum().item())
            total = int(nan_any.numel())
            print(f"[SeManSanity] {tag} nan={nan_cnt}/{total}")

        with torch.no_grad():
            # Pass A: fixed colors_precomp + fixed opacities
            rendervar_const = {
                "means3D": seman_rendervar["means3D"],
                "colors_precomp": torch.zeros_like(seman_rendervar["colors_precomp"]),
                "rotations": seman_rendervar["rotations"],
                "opacities": torch.full_like(seman_rendervar["opacities"], 0.01),
                "scales": seman_rendervar["scales"],
                "means2D": seman_rendervar["means2D"],
            }
            sem_const, _, = sem_renderer(raster_settings=sparse_cam)(**rendervar_const)
            _nan_stats("const_inputs", sem_const)

            # Pass B: zero cls_ids (cam_zero built by caller)
            sem_zero, _, = sem_renderer(raster_settings=cam_zero)(**seman_rendervar)
            _nan_stats("zero_cls_ids", sem_zero)
    except Exception as exc:
        print(f"[SeManSanity] failed ({exc})")


def log_seman_loss_stats(seman_im: torch.Tensor, seman_tgt: torch.Tensor) -> None:
    if not env_flag("ACTIVE_SGM_DEBUG_SEMAN_LOSS", False):
        return
    max_logs = env_int("ACTIVE_SGM_DEBUG_SEMAN_LOSS_MAX", 20)
    mode = env_str("ACTIVE_SGM_DEBUG_SEMAN_LOSS_MODE", "nonfinite").lower()
    if _counter_val("seman_loss_stats") >= max_logs:
        return

    def _stats(t: torch.Tensor) -> Tuple[int, float, float, float]:
        finite = torch.isfinite(t)
        bad = int((~finite).sum().item())
        if finite.any():
            vals = t[finite]
            vmin = float(vals.min().item())
            vmax = float(vals.max().item())
            vmean = float(vals.mean().item())
        else:
            vmin = vmax = vmean = float("nan")
        return bad, vmin, vmax, vmean

    bad_seman, vmin_s, vmax_s, vmean_s = _stats(seman_im)
    bad_tgt, vmin_t, vmax_t, vmean_t = _stats(seman_tgt)
    any_bad = (bad_seman > 0) or (bad_tgt > 0)
    if mode == "nonfinite" and not any_bad:
        return
    _bump_counter("seman_loss_stats")
    print(
        "[SeManLossDebug] "
        f"seman_im shape={tuple(seman_im.shape)} bad={bad_seman} min={vmin_s:.6f} max={vmax_s:.6f} mean={vmean_s:.6f} "
        f"tgt shape={tuple(seman_tgt.shape)} bad={bad_tgt} min={vmin_t:.6f} max={vmax_t:.6f} mean={vmean_t:.6f}"
    )


def log_seman_nan_coverage(
    seman_im: torch.Tensor,
    presence_sil_mask: torch.Tensor,
    mask: torch.Tensor,
    curr_data: Dict[str, torch.Tensor],
    iter_time_idx: Optional[int] = None,
) -> None:
    if not env_flag("ACTIVE_SGM_DEBUG_SEMAN_NAN", False):
        return
    max_logs = env_int("ACTIVE_SGM_DEBUG_SEMAN_NAN_MAX", 10)
    if _counter_val("seman_nan") >= max_logs:
        return
    try:
        nan_any = torch.isnan(seman_im).any(dim=0)
        nan_cnt = int(nan_any.sum().item())
        total = int(nan_any.numel())
        depth_valid = (curr_data["depth"] > 0)
        nan_in_depth = int((nan_any & depth_valid).sum().item())
        nan_out_depth = int((nan_any & ~depth_valid).sum().item())
        nan_in_sil = int((nan_any & presence_sil_mask).sum().item())
        nan_out_sil = int((nan_any & ~presence_sil_mask).sum().item())
        mask_hw = mask[0] if mask.dim() == 3 else mask
        nan_in_mask = int((nan_any & mask_hw).sum().item())
        nan_out_mask = int((nan_any & ~mask_hw).sum().item())
        crop_msg = ""
        if "crop_mask" in curr_data.keys():
            crop_mask = curr_data["crop_mask"]
            nan_in_crop = int((nan_any & crop_mask).sum().item())
            nan_out_crop = int((nan_any & ~crop_mask).sum().item())
            crop_msg = f" nan_in_crop={nan_in_crop} nan_out_crop={nan_out_crop}"
        prefix = "[SeManNaNDebug]"
        if iter_time_idx is not None:
            prefix = f"[SeManNaNDebug] iter={iter_time_idx}"
        print(
            f"{prefix} "
            f"nan={nan_cnt}/{total} "
            f"nan_in_depth={nan_in_depth} nan_out_depth={nan_out_depth} "
            f"nan_in_sil={nan_in_sil} nan_out_sil={nan_out_sil} "
            f"nan_in_mask={nan_in_mask} nan_out_mask={nan_out_mask}{crop_msg}"
        )
        _bump_counter("seman_nan")
    except Exception as exc:
        print(f"[SeManNaNDebug] failed ({exc})")


def log_nonfinite_loss(loss: torch.Tensor) -> None:
    if not env_flag("ACTIVE_SGM_DEBUG_SEMAN_LOSS", False):
        return
    if not torch.isfinite(loss):
        print(f"[SeManLossDebug] loss is non-finite: {loss}")
