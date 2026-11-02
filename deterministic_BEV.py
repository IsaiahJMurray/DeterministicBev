from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

ROOT = Path(__file__).resolve().parent
CAMERAS = set(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])

class DeterministicBEV:
    def __init__(self, dataset, loader, args):
        self.dataset = dataset
        self.loader = loader
        self.args = args
        self.nusc = NuScenes(version='v1.0-mini', dataroot=str(ROOT / 'data/nuscenes'), verbose=True)

    # Returns a list of rays IN THE CAMERA FRAME, from pixel coordinates.
    def camera_uvs_to_rays(self, camera, input_uv, scene_record):
        cam_token = scene_record['data'][camera]
        cam_data = self.nusc.get('sample_data', cam_token)
        calibrated_sensor = self.nusc.get(
            'calibrated_sensor',
            cam_data['calibrated_sensor_token'],
        )

        intrinsics = np.asarray(calibrated_sensor['camera_intrinsic'], dtype=float)
        pixels = np.column_stack([
            np.asarray(input_uv, dtype=float),
            np.ones(len(input_uv)),
        ])

        # Shape: (N, 3)
        rays = pixels @ np.linalg.inv(intrinsics).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)

        return rays

    def nearest_direction_points(rays, points, max_angle_deg=2.0):
        point_norms = np.linalg.norm(points, axis=1)
        point_directions = points / point_norms[:, None]

        scores = point_directions @ rays.T
        best_indices = np.argmax(scores, axis=0)
        best_scores = scores[best_indices, np.arange(len(rays))]

        matched = best_scores >= np.cos(np.deg2rad(max_angle_deg))

        selected_points = np.full_like(rays, np.nan)
        depths = np.full(len(rays), np.nan)

        selected_points[matched] = points[best_indices[matched]]
        depths[matched] = np.sum(
            selected_points[matched] * rays[matched],
            axis=1,
        )

        return selected_points, depths

    # This function takes a list of rays and projects them into the lidar frame.
    def rays_to_ego_depth_via_lidar(self, rays, scene_record, ref_camera):
        lidar_token = scene_record['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token']).T

        pc, times = LidarPointCloud.from_file_multisweep(
            self.nusc, 
            sample_rec=scene_record, 
            chan='LIDAR_TOP', 
            ref_chan=ref_camera, # Automatically transforms into this camera's frame
            nsweeps=10
        )        
