# sem-map との比較実験用メイン config（全シーン共通・env 駆動）。
# 各シーンの ActiveSem.py を継承し、340x600 variant を指す。scene は env ACTIVESGM_SCENE。
# start_c2w は env ACTIVESGM_START_C2W(.npy パス)から注入（sem-map の初期ポーズ camera_poses.npz の c2w[0], RUB）。
# baseline config は一切編集しないオーバーレイ方式。
import os
import numpy as np

# 継承元はシーン毎の ActiveSem.py（この config からの相対 = configs/MP3D/{scene}/ActiveSem.py）。
# → general.scene / bbox_bound / semantic 設定などをそのシーンの値で継承。
# 注意: mmengine は _base_ を静的に(隔離名前空間で)eval するため builtin __import__ で env を参照する。
_base_ = __import__("os").path.join(
    __import__("os").environ.get("ACTIVESGM_SCENE", "GdvgFV5R1Z5"), "ActiveSem.py")

# 解像度 340x600 の比較 variant を指す（CWD=submodule ルート基準の相対パス）
sim = dict(
    habitat_cfg = "configs/MP3D/habitat_cmp340.py",
)
slam = dict(
    room_cfg = "configs/MP3D/mp3d_splatam_s_cmp340.py",
)

# sem-map の初期ポーズ注入（指定時のみ。未指定なら継承元 ActiveSem.py の start_c2w を使用）
_start_c2w_path = os.environ.get("ACTIVESGM_START_C2W", "")
if _start_c2w_path:
    slam["start_c2w"] = np.load(_start_c2w_path).astype(np.float32)

# 単一GPU等で意味セグ用デバイスを変えたい場合（未指定なら継承元の値=cuda:1）
_sem_dev = os.environ.get("ACTIVESGM_SEMANTIC_DEVICE", "")
if _sem_dev:
    slam["semantic_device"] = _sem_dev
