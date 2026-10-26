"""Run the bundled official BEVFormer Tiny checkpoint on nuScenes mini.

Timings include the model call, host-to-GPU transfer and postprocessing.
Dataset loading is reported separately. Use --frames 0 for all validation frames.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'BEVFormer'))

import mmcv
from mmcv.parallel import MMDataParallel
from mmcv import Config
import numpy as np
import torch

import projects.mmdet3d_plugin  # Register BEVFormer and its dataset/pipeline.
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

CHECKPOINT_SHA256 = '7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12'

# AI Gen debug will be replaced
def plot_prediction(result, info, classes, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    fig = plt.figure(figsize=(18, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.6])
    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    for i, camera in enumerate(cameras):
        ax = fig.add_subplot(grid[i // 3, i % 3])
        with Image.open(info['cams'][camera]['data_path']) as image:
            image.thumbnail((600, 338))
            ax.imshow(image)
        ax.set_title(camera.removeprefix('CAM_').replace('_', ' '), fontsize=10)
        ax.axis('off')
    ax = fig.add_subplot(grid[:, 3])
    prediction = result['pts_bbox']
    corners = prediction['boxes_3d'].corners.detach().cpu().numpy()
    scores = prediction['scores_3d'].detach().cpu().numpy()
    labels = prediction['labels_3d'].detach().cpu().numpy()
    colors = plt.get_cmap('tab10')
    used = set()
    for box, score, label in zip(corners, scores, labels):
        if score < 0.3:
            continue
        ring = box[[0, 3, 7, 4, 0], :2]
        name = classes[int(label)]
        ax.plot(ring[:, 0], ring[:, 1], color=colors(int(label)),
                label=name if name not in used else None, linewidth=1.3)
        used.add(name)
    ax.scatter([0], [0], marker='^', color='black', s=60, label='Vehicle')
    ax.set(xlim=(-51.2, 51.2), ylim=(-51.2, 51.2), aspect='equal',
           xlabel='LiDAR x (m)', ylabel='LiDAR y (m)',
           title='Camera-only BEV predictions\nconfidence ≥ 0.30')
    ax.grid(alpha=0.2)
    ax.legend(loc='upper right', fontsize=7)
    fig.suptitle('BEVFormer Tiny · nuScenes mini · ' + info['token'], fontsize=12)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def main():
    # These args are for benchmarking
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    # Load the BEVFORMER config.
    cfg = Config.fromfile(str(ROOT / 'BEVFormer/projects/configs/bevformer/bevformer_tiny.py'))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    # Add the mini dataset to the config.
    config = cfg.data.test.copy()
    config.update(data_root=str(ROOT / 'data/nuscenes'), ann_file=str(ROOT / 'data/nuscenes/nuscenes_infos_temporal_val.pkl'),test_mode=True)
    config.pop('samples_per_gpu', None)
    dataset = build_dataset(config)

    # See how many frames the users running.
    count = min(args.frames or len(dataset), len(dataset))
    if args.eval and count != len(dataset):
        parser.error('--eval requires --frames 0 or the complete validation split')
    print(f'Running {count} frames', flush=True)

    # This is the dataset's loader. We only use one worker because im scared of parallelism in a model this old.
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    # Load our model!
    checkpoint_path = ROOT / 'BEVFormer/ckpts/bevformer_tiny_epoch_24.pth'
    with checkpoint_path.open('rb') as handle:
        checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    weights = checkpoint['state_dict']
    if any(key.startswith('module.') for key in weights):
        weights = OrderedDict((key.removeprefix('module.'), value) for key, value in weights.items())
    model.load_state_dict(weights, strict=True)

    # Drop these huge variables (my computer is stressing on memory).
    del weights, checkpoint

    # Make sure classes are a;igned
    model.CLASSES = dataset.CLASSES

    # Use the gpu to run the model.
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    model = MMDataParallel(model.cuda(), device_ids=[0]).eval()
    torch.cuda.reset_peak_memory_stats()

    print('Model initialized and ready to run', flush=True)

    # This loop runs the model on the dataset and times it
    outputs, timings, end_to_end = [], [], []
    iterator = iter(loader)
    for index in range(count):
        start = time.perf_counter()
        data = next(iterator)

        # Synchronization step
        torch.cuda.synchronize()
        infer_start = time.perf_counter()

        # We aren't training, so are not calculating gradients.
        # I Think this is a cool way to save memory.
        with torch.no_grad():
            # Run the model!
            result = model(return_loss=False, rescale=True, **data)

        # After the model is done, we sync again to make sure the GPU is done 
        torch.cuda.synchronize()
        finish = time.perf_counter()
        outputs.extend(result)

        # Skips warmup frames.
        if index >= args.warmup:
            timings.append((finish - infer_start) * 1000)
            end_to_end.append((finish - start) * 1000)
        print(f'Frame {index + 1}/{count} done in {(finish - start) * 1000:.1f} ms', flush=True)

    # Save our outputs.
    destination = ROOT / 'outputs'
    destination.mkdir(exist_ok=True)
    for sample in outputs:
        prediction = sample['pts_bbox']
        # Save the boxes, scores and labels to a pkl file.
        report = dict(torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), mmcv=mmcv.__version__,
                  split='v1.0-mini / mini_val', frames=count,
                  warmup_frames=min(args.warmup, count), measured_frames=len(timings),
                  model_mean_ms=float(np.mean(timings)) if timings else None,
                  model_median_ms=float(np.median(timings)) if timings else None,
                  model_p95_ms=float(np.percentile(timings, 95)) if timings else None,
                  end_to_end_mean_ms=float(np.mean(end_to_end)) if end_to_end else None,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                  checkpoint_sha256=checksum,
                  timing_scope='Model call including transfer and postprocessing; end-to-end also includes dataset loading.')
    # Save the data to disk
    mmcv.dump(outputs, str(destination / 'predictions.pkl'))
    (destination / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    plot_prediction(outputs[0], dataset.data_infos[0], dataset.CLASSES,
                    destination / 'prediction.png')
    # Evaluate the predictions if requested.
    if args.eval:
        metrics = dataset.evaluate(outputs, jsonfile_prefix=str(destination / 'evaluation'))
        (destination / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Outputs:', destination)


if __name__ == '__main__':
    main()
