# 比較実験用 habitat config: 各シーンの habitat.py を継承し、解像度のみ 340x600 (sem-map 一致) に上書き。
# scene は環境変数 ACTIVESGM_SCENE で指定（未指定時は GdvgFV5R1Z5）。baseline の habitat.py は非編集。
import os
import numpy as np

# 継承元は env で指定したシーンの habitat.py（この config からの相対 = configs/MP3D/{scene}/habitat.py）。
# 注意: mmengine は _base_ を静的に(隔離名前空間で)eval するためローカル変数やimport済みosは使えない。
#       builtin の __import__ でのみ環境変数を参照する。
_base_ = __import__("os").path.join(
    __import__("os").environ.get("ACTIVESGM_SCENE", "GdvgFV5R1Z5"), "habitat.py")

# 解像度 340x600 / focal 300 → hfov=90°, vfov≈59°（sem-map と同一視野・同一 intrinsics）
fov = lambda size, focal: np.rad2deg(np.arctan((size / 2) / focal)) * 2
camera = dict(
    pinhole = dict(
        resolution_hw = [340, 600],
        fov = (fov(340, 300), fov(600, 300)),  # h, w
    )
)
