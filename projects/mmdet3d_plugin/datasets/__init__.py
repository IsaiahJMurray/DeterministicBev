from .nuscenes_dataset import CustomNuScenesDataset
try:
    from .nuscenes_dataset_v2 import CustomNuScenesDatasetV2
except ImportError as _e:
    # Guarded by patches/bevformer/10_datasets_init_guard.sh
    # Reason: v2 pulls dd3d.datasets.nuscenes -> Detectron2 (un-installed in v5)
    import warnings
    warnings.warn(f'CustomNuScenesDatasetV2 unavailable: {_e}')

from .builder import custom_build_dataset
__all__ = [
    'CustomNuScenesDataset',
    'CustomNuScenesDatasetV2',
]
