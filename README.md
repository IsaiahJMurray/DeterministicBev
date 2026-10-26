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
