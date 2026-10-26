from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

ROOT = Path(__file__).resolve().parent
CAMERAS = set(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])

class DeterministicBEV:
    def __init__(self, dataset, loader, args):
        self.dataset = dataset
        self.loader = loader
        self.args = args
        self.nusc = NuScenes(version='v1.0-mini', dataroot=str(ROOT / 'data/nuscenes'), verbose=True)

    def camera_uv_to_vec(self, camera, input_uv, scene_record):
        cam_token = scene_record['data'][camera]
        cam_data = self.nusc.get('sample_data', cam_token)
        calibrated_sensor = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        # This is the camera intrinsics matrix.
        intrinsics = np.array(calibrated_sensor['camera_intrinsic'])
        inv_intrinsics = np.linalg.inv(intrinsics)

        # This is the camera pos in car coordinates.
        translation_ego = np.array(calibrated_sensor['translation'])

        # This is the camera rotation in car coordinates.
        rotation_quat = calibrated_sensor['rotation']

        # This is the camera to ego transformation matrix.
        R_cam2ego = Quaternion(rotation_quat).rotation_matrix

        # NGL copilot autocompleted this
        for (u, v) in input_uv:
            pixel_homog = np.array([u, v, 1.0])
            ray_cam = inv_intrinsics @ pixel_homog
            ray_ego = R_cam2ego @ ray_cam
            ray_ego /= np.linalg.norm(ray_ego)
            yield ray_ego
