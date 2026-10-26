import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

CAMERAS = set(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])

class DeterministicBEV:
    def __init__(self, dataset, loader, args):
        self.dataset = dataset
        self.loader = loader
        self.args = args

    def camera_to_ego(self, camera, input_points):
        cam_token = 