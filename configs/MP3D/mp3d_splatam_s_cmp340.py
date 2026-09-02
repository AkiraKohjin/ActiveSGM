# 比較実験用 SplaTAM room config: mp3d_splatam_s.py の config を読み込み、
# SLAM 解像度を 340x600 に、dataset_sample の basedir を環境変数で差し替える。baseline は非編集。
#
# 注意: room_cfg は SplatamOurs.__init__ が SourceFileLoader で生モジュールとして読み、
#       .config を直接取得する（mmengine の _base_ 継承は効かない）。
#       そのため base の config を明示的に読み込んでから上書きし、完全な config を提供する。
import os
from importlib.machinery import SourceFileLoader

_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp3d_splatam_s.py")
config = SourceFileLoader("mp3d_splatam_s_base", _base_path).load_module().config

# data(dataset_sample) だけ上書き。他のキー(map_every 等)は base のまま保持。
config["data"] = dict(config["data"])
config["data"].update(
    basedir = os.environ.get("ACTIVESGM_SAMPLE_DIR", "./data/mp3d_sim_nvs"),
    desired_image_height = 340,
    desired_image_width = 600,
    tracking_image_height = 340,
    tracking_image_width = 600,
    # densification は baseline と同じ「desired の半分」比を維持（680x1200→340x600 に対応し 340x600→170x300）。
    # desired と一致させると seperate_densification_res=False 経路の不具合(densify_dataset_sample 未設定)を踏むため。
    densification_image_height = 170,
    densification_image_width = 300,
)
