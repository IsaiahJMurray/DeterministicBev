from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version="v1.0-mini", dataroot="v1.0-mini", verbose=False)
# print(len(nusc.sample))   # expect 404
sample = nusc.get("sample", nusc.scene[0]["first_sample_token"]) # a specific moment in time captured


import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

# read the calibration for CAM_FRONT

sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"]) # sample data
cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]) # calibrated sensor

K = np.array(cs["camera_intrinsic"])                                          
T_ego_cam = transform_matrix(cs["translation"], Quaternion(cs["rotation"]))   # camera's pose in ego frame
T_cam_ego = np.linalg.inv(T_ego_cam)                                          # ego's frame in the cam frame

print("K =\n", K)
print("camera position in ego frame (x fwd, y left, z up):", cs["translation"])