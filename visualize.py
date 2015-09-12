import numpy as np
from visual import *

# for swapping between OpenCV and VPython coordinates
coord_swap = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

# initialize landmark geometry for drawing
# face_spheres_3d = []
# for _ in range(68):
#     face_spheres_3d.append(sphere(pos=(0, 0, 0), radius=1, color=color.red))
#
# # init eyeball geometry
# eyeballs_3d = [sphere(pos=(0, 0, 0), radius=12, color=color.white, opacity=0.5),
#                sphere(pos=(0, 0, 0), radius=12, color=color.white, opacity=0.5)]
# gaze_vecs_3d = [arrow(pos=(0,0,0), axis=(1,0,0), shaftwidth=0.5, color=color.blue),
#                 arrow(pos=(0,0,0), axis=(1,0,0), shaftwidth=0.5, color=color.blue)]
#
# head_pose_vec_3d = arrow(pos=(0,0,0), axis=(5,0,0), shaftwidth=10, color=color.red)

# position 3d face pts
# for i, pt in enumerate(face_pts_3d):
#     face_spheres_3d[i].pos = coord_swap.dot(np.array(pt).T)

def visualize(pts_3d, radius=1, color=color.white):

    scene.autocenter = True

    for pt in pts_3d:
        sphere(pos=coord_swap.dot(np.array(pt)), radius=radius, color=color, opacity=0.5)
