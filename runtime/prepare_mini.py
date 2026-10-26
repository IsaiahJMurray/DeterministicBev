# This script preps the mini nuScenes dataset for feeding into BEVFormer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'BEVFormer'))

from tools.data_converter.nuscenes_converter import create_nuscenes_infos

if __name__ == '__main__':
    create_nuscenes_infos(
        root_path=str(ROOT / 'data/nuscenes'),
        out_path=str(ROOT / 'data/nuscenes'),
        can_bus_root_path=str(ROOT / 'data'),
        info_prefix='nuscenes',
        version='v1.0-mini',
        max_sweeps=10,
    )
