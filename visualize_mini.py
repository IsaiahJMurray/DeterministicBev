"""Render camera views and a simple LiDAR BEV from nuScenes mini."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "nuscenes"
OUTPUT_ROOT = ROOT / "visualizations"
CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)


def main():
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)
    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"nuScenes mini — {scene['name']}: {scene['description']}", fontsize=15, y=0.985)

    for index, camera in enumerate(CAMERAS):
        row, col = divmod(index, 3)
        x = 0.02 + col * 0.32
        y = 0.70 - row * 0.28
        ax = fig.add_axes([x, y, 0.30, 0.245])
        image_token = sample["data"][camera]
        image_path = DATA_ROOT / nusc.get("sample_data", image_token)["filename"]
        ax.imshow(Image.open(image_path))
        ax.set_title(camera)
        ax.axis("off")

    ax = fig.add_axes([0.34, 0.035, 0.32, 0.35])
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
    points = LidarPointCloud.from_file(lidar_path).points
    scatter = ax.scatter(
        points[0], points[1], c=points[2], s=0.35, cmap="viridis", alpha=0.8,
        vmin=-2.5, vmax=2.5, linewidths=0,
    )
    for box in boxes:
        corners = box.corners()
        outline = np.r_[0:4, 0]
        ax.plot(corners[0, outline], corners[1, outline], color="tomato", linewidth=1)
    ax.scatter([0], [0], marker="^", s=50, c="red", edgecolors="white", linewidths=0.5)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.set_xlabel("Forward (m)")
    ax.set_ylabel("Left (m)")
    ax.set_title("Top-down LiDAR with annotated 3D boxes (red); point color shows height")
    ax.grid(alpha=0.2)
    colorbar_ax = fig.add_axes([0.68, 0.06, 0.012, 0.28])
    fig.colorbar(scatter, cax=colorbar_ax, label="Height (m)")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{scene['name']}_camera_and_bev.png"
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
