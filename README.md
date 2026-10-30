# Project 3: BEVFormer + nuScenes

## Environment

Already installed: Conda `bevformer`, Python 3.11, PyTorch 2.8 / CUDA 12.8.

```bash
cd "/home/isa/Documents/Comp Robo/Project3"
source activate.sh
```

## Start

```bash
python run_mini.py --frames 10 --warmup 3
```

Full mini validation + benchmark:

```bash
python run_mini.py --frames 0 --warmup 5 --eval
```

Results: `outputs/` (prediction image, timings, detections, metrics).

[Environment details](runtime/README.md)

---

# BEV Cam (Jack)

This section is for the camera-only deterministic BEV (`deterministic_bev_cam.py`). It runs on CPU and does not need the BEVFormer environment above.

## Setup (macOS, one time)

Run from the project folder. Python 3.12 is used because `nuscenes-devkit` may not have wheels for newer Python versions.

```bash
cd /Users/welchj/Desktop/DeterministicBev
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install nuscenes-devkit pyquaternion opencv-python matplotlib
```

## Run

Activate the venv in each new terminal, then run from the project folder (the dataset path `v1.0-mini` is relative):

```bash
cd /Users/welchj/Desktop/DeterministicBev
source .venv/bin/activate
python deterministic_bev_cam.py
```

Expected output: `404`, then `(900, 1600, 3)` for each of the 6 cameras.
