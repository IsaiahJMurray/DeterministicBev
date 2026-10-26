# Native BEVFormer Tiny environment

This setup targets pretrained BEVFormer Tiny inference on the RTX PRO 2000
Blackwell laptop GPU (compute capability 12.0). Full training is not validated.
The separate `../.venv` remains the environment for the dataset visualizer.

## Use the environment

```bash
cd "/home/isa/Documents/Comp Robo/Project3"
source activate.sh
python run_mini.py --frames 10 --warmup 3
```

For the complete mini validation split and exploratory detection metrics:

```bash
python run_mini.py --frames 0 --warmup 5 --eval
```

Results are written to `../outputs/`: a camera/BEV prediction image,
`predictions.pkl`, and `benchmark.json`. A subsequent run overwrites these files.
Detection evaluation additionally writes `metrics.json` and its evaluation files.
The mini split is for experiments; its metrics are not the published full
nuScenes validation benchmark.

Timings exclude dataset loading for `model_*_ms`, but include the model's input
transfer and postprocessing. `end_to_end_mean_ms` also includes dataset loading.
Warmup frames are excluded from latency statistics. Tests use batch size one.

## Stack and compatibility

- Python 3.11 in the `bevformer` Conda environment.
- PyTorch 2.8.0 + CUDA 12.8; torchvision 0.23.0.
- CUDA compiler 12.8.93, installed from NVIDIA's Conda channel.
- MMCV-full 1.7.2, compiled locally with `TORCH_CUDA_ARCH_LIST=12.0`.
- MMDetection 2.28.2, MMSegmentation 0.30.0, MMDetection3D 0.17.1.
- NumPy 1.26.4, numba 0.58.1, nuScenes devkit 1.2.0.

`constraints.txt` and `requirements.txt` record selected package versions. The
setup uses compatibility patches from
[nabe2030/bevformer-blackwell](https://github.com/nabe2030/bevformer-blackwell),
adapted for this GPU's architecture. Its repository is saved in `compat/`.
MMDetection3D's source is linked at `mmdetection3d/`; its actual location is in
the Conda environment's `src/` directory to avoid compiler issues with spaces in
the project path. Modern runtime version constraints replace its obsolete
NumPy/numba/networkx/trimesh requirements. `patch_mmcv.py` records the GPU stream
API adjustment. The optional DD3D/Detectron2 paths are excluded from imports;
the original Tiny model and its active data pipeline are retained.

`activate.sh` removes inherited ROS Python paths from this shell so Python 3.12
ROS packages do not enter the Python 3.11 environment. It also sets header paths
for compiling CUDA extensions with the minimal toolkit.

`run_mini.py` checks the bundled official checkpoint's SHA-256 before loading
its legacy checkpoint format. It refuses a changed checkpoint. Only load
pickled model/result files from trusted sources.

## Dataset preparation

```bash
python runtime/fetch_canbus_mini.py
python runtime/prepare_mini.py
```

The CAN bus downloader uses HTTP range reads against the official nuScenes
archive and saves only the ten mini-scene pose JSON files. Dataset preparation
uses BEVFormer's original temporal converter and the real CAN bus poses.
No placeholder motion data is generated.

Installation and build logs are in `logs/`.

## Verified baseline — October 25, 2026

The complete 81-frame mini validation run succeeded, including detection
evaluation. With 5 warmup frames and 76 measured frames, mean model latency was
130.4 ms and mean end-to-end latency was 292.9 ms. Peak allocated GPU memory was
625 MiB. Mini mAP was 0.2647 and NDS was 0.3252. These are exploratory mini-split
results, not the published full validation benchmark. Installed dependency
metadata also passed `pip check`. Exact installed Python versions are recorded
in `pip-freeze.txt`.
