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

# fetching transformation ref frames
K = np.array(cs["camera_intrinsic"]) # intrinsic lens matrix (already calibrated)
T_ego_cam = transform_matrix(cs["translation"], Quaternion(cs["rotation"]))   # camera's pose in ego frame
T_cam_ego = np.linalg.inv(T_ego_cam)                                          # ego's frame in the cam frame

# print("K =\n", K)
# print("camera position in ego frame (x fwd, y left, z up):", cs["translation"])

# build the BEV grid in ego coords in meters
extent, res = 50.0, 0.1                   # cover +-50 m, 10 cm per pixel
size = int(2 * extent / res)              # 1000 x 1000 pixels

cols, rows = np.meshgrid(np.arange(size), np.arange(size))
# arrays to correspond the pixel coords in BEV image with the ego-frame coords in meters 
# following same convention as nuscence dataset does (left is positive)
grid_x = extent - (rows + 0.5) * res      # forward distance (m); top row = +50 m ahead
grid_y = extent - (cols + 0.5) * res      # left distance (m); left column = +50 m to the left

# print(size, grid_x[0, 0], grid_y[0, 0], grid_x[-1, -1], grid_y[-1, -1])

# ========== PROCESSING ==============
import cv2
img = cv2.cvtColor(cv2.imread("v1.0-mini/" + sd["filename"]), cv2.COLOR_BGR2RGB)


# show othe raw image
import matplotlib.pyplot as plt         
plt.imshow(img)
plt.title("CAM_FRONT")
plt.axis("off")
plt.show() 


# project the ground grid into the front camera, flattening (assuming all are grounds for now)
pts = np.stack([grid_x.ravel(), grid_y.ravel(),
                np.zeros(grid_x.size), np.ones(grid_x.size)])   # 4 x N : (x, y, z=0, 1), last one just for easier matrix multiplication
cam = (T_cam_ego @ pts)[:3]               # the same points in the camera frame (drops 4th)
uvw = K @ cam                             # apply the lens matrix
depth = uvw[2]                            # distance in front of the camera
u = uvw[0] / np.maximum(depth, 1e-6)      # pixel column in the image
v = uvw[1] / np.maximum(depth, 1e-6)      # pixel row in the image

h, w = img.shape[:2]
valid = (depth > 0.1) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) 

valid = valid.reshape(size, size)
# print("fraction of BEV seen by CAM_FRONT:", valid.mean())
# print("front half:", valid[:500].mean(), " back half:", valid[500:].mean())


# Sample the image colors into the BEV
map_x = np.where(valid, u.reshape(size, size), -1).astype(np.float32)   # image column for each BEV pixel
map_y = np.where(valid, v.reshape(size, size), -1).astype(np.float32)   # image row for each BEV pixel
bev_front = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

print("BEV shape:", bev_front.shape)    # expect (1000, 1000, 3)

plt.imshow(bev_front)
plt.title("CAM_FRONT -> BEV")
plt.axis("off")
plt.show()
