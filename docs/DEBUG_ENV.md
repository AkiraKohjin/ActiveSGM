# ActiveSGM Debug Environment Variables

This document lists debug-related environment variables used in `semsplatam.py` and
`modified_ver/splatam/splatam.py`. These flags are **off by default**.

## Semantic Visualization (OneFormer)
- `ACTIVE_SGM_DEBUG_SEM` (bool)
  - Save per-frame semantic debug outputs (RGB, IDs, colorized, overlay, confidence, histogram).
- `ACTIVE_SGM_DEBUG_SEM_EVERY` (int, default: `1`)
  - Save every N frames.
- `ACTIVE_SGM_DEBUG_SEM_MAX` (int, default: `-1`)
  - Max frame index to save (`-1` = no limit).

## Semantic Tensor Stats
- `ACTIVE_SGM_DEBUG_SEMAN_STATS` (bool)
  - Log semantic tensor stats (min/max/mean, shape, contiguity).
- `ACTIVE_SGM_DEBUG_SEMAN_STATS_EVERY` (int, default: `1`)
  - Log every N frames.
- `ACTIVE_SGM_DEBUG_SEMAN_STATS_MAX` (int, default: `-1`)
  - Max frame index to log (`-1` = no limit).

## GPU / Transfer Diagnostics
- `ACTIVE_SGM_DEBUG_SEMAN_P2P` (bool)
  - Print device info and GPU peer-access capability + copy test.
- `ACTIVE_SGM_DEBUG_SEMAN_TRANSFER` (bool)
  - Log NaN/Inf stats before/after semantic tensor transfers.
- `ACTIVE_SGM_DEBUG_SEMAN_TRANSFER_MAX` (int, default: `5`)
  - Max transfer logs.
- `ACTIVE_SGM_DEBUG_SEMAN_TRANSFER_SYNC` (bool)
  - Force CUDA sync before transfer (determinism/debug).
- `ACTIVE_SGM_SEMAN_CPU_HOP` (bool)
  - Force semantic tensors to go GPU -> CPU -> GPU (avoid P2P corruption).
- `ACTIVE_SGM_DEBUG_SEMAN_SYNC` (bool)
  - Print semantic tensor shape/device and synchronize.
- `ACTIVE_SGM_DEBUG_SEMAN_CPU_CHECK` (bool)
  - Check CPU copy for NaN/Inf.

## OneFormer Output Logging
- `ACTIVE_SGM_DEBUG_ONEFORMER` (bool)
  - Log OneFormer logits stats.
- `ACTIVE_SGM_DEBUG_ONEFORMER_MAX` (int, default: `20`)
  - Max logs.
- `ACTIVE_SGM_DEBUG_ONEFORMER_MODE` (string, default: `nonfinite`)
  - `nonfinite` logs only if NaN/Inf present; any other value logs always.

## Rendered Semantic Debug (SemSplatam.render_semantic)
- `ACTIVE_SGM_DEBUG_RENDER_SEM` (bool)
  - Log renderer input/output stats for semantic render.
- `ACTIVE_SGM_DEBUG_RENDER_SEM_MAX` (int, default: `20`)
  - Max logs.
- `ACTIVE_SGM_DEBUG_RENDER_SEM_MODE` (string, default: `nonfinite`)
  - `nonfinite` logs only if NaN/Inf present; any other value logs always.

## Semantic Renderer Inputs / Determinism
- `ACTIVE_SGM_DEBUG_SEMAN_RENDERVAR` (bool)
  - Log semantic renderer input stats.
- `ACTIVE_SGM_DEBUG_SEMAN_RENDERVAR_MAX` (int, default: `10`)
  - Max logs.
- `ACTIVE_SGM_DEBUG_SEMAN_RENDERVAR_MODE` (string, default: `nonfinite`)
  - `nonfinite` logs only if NaN/Inf present; any other value logs always.
- `ACTIVE_SGM_DEBUG_SEM_REPEAT` (bool)
  - Re-render once and compare outputs (determinism check).
- `ACTIVE_SGM_DEBUG_SEM_REPEAT_MAX` (int, default: `1`)
  - Max repeat logs.
- `ACTIVE_SGM_DEBUG_SEM_SANITY` (bool)
  - Sanity render with fixed inputs + zero cls_ids.
- `ACTIVE_SGM_DEBUG_SEM_SANITY_MAX` (int, default: `1`)
  - Max sanity logs.

## Semantic Loss / NaN Coverage
- `ACTIVE_SGM_DEBUG_SEMAN_LOSS` (bool)
  - Log semantic loss inputs/outputs and non-finite loss.
- `ACTIVE_SGM_DEBUG_SEMAN_LOSS_MAX` (int, default: `20`)
  - Max logs.
- `ACTIVE_SGM_DEBUG_SEMAN_LOSS_MODE` (string, default: `nonfinite`)
  - `nonfinite` logs only if NaN/Inf present; any other value logs always.
- `ACTIVE_SGM_DEBUG_SEMAN_NAN` (bool)
  - Log NaN coverage stats in semantic render output.
- `ACTIVE_SGM_DEBUG_SEMAN_NAN_MAX` (int, default: `10`)
  - Max logs.

## Semantic Init / Background / Entropy
- `ACTIVE_SGM_DEBUG_INIT_SEM` (bool)
  - Log semantic_logits stats during initialization.
- `ACTIVE_SGM_DEBUG_SEM_BG` (bool)
  - Log semantic background (bg) channel length checks.
- `ACTIVE_SGM_DEBUG_SEM_BG_MAX` (int, default: `5`)
  - Max bg logs.
- `ACTIVE_SGM_DEBUG_ENTROPY_FUNC` (bool)
  - Log NaN/Inf in entropy inputs.
- `ACTIVE_SGM_DEBUG_ENTROPY_FUNC_MAX` (int, default: `20`)
  - Max entropy logs.

## Mapping Param Debug
- `ACTIVE_SGM_DEBUG_PARAM_SEM` (bool)
  - Log semantic parameter stats + gradients during mapping.
- `ACTIVE_SGM_DEBUG_PARAM_SEM_MAX` (int, default: `20`)
  - Max logs.
