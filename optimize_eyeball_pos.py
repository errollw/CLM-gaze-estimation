import cv2
import numpy as np
import pickle
import os
import geom_utils

import scipy.optimize

fx, fy = 757.186370667, 757.260080183
cx, cy = 412.671083627, 272.671560372
camera_mat = np.array([[fx, 0,  cx],
                       [0,  fy, cy],
                       [0,  0,  0]], dtype=float)

path = "gaze_calibration_imgs"

def get_target_gaze_pos(pt):

    pt = int(pt)-1
    i, j = pt / 3, pt - (pt/3)*3

    # print pt,i,j

    return np.array([1720/2 * j + 100, 1000/2 * i + 100])

def calc_gaze_pos(face, eye0, eye1, offset=[0, -2, 0, 1]):

    intersections = []
    for i in range(2):

        iris_pts = eye0.iris_pts_3d if i == 0 else eye1.iris_pts_3d
        pupil = np.mean(iris_pts, axis=0)
        ray_dir = pupil / float(np.linalg.norm(pupil))

        # position eyeball in 3d
        pose_transform = face.get_rotation_transform()
        if i == 1: offset = np.array([-1, 1, 1, 1]) * offset
        eyeball_offset = pose_transform.dot(np.array(offset))
        eyeball_3d_pos = (np.array(face.pts_3d[36+i*6])+np.array(face.pts_3d[39+i*6]))/2.0 + eyeball_offset[:3]

        try:
            P = geom_utils.ray_sphere_intersect((0,0,0), ray_dir, eyeball_3d_pos, 12)
        except geom_utils.NoIntersection:
            continue

        gaze_vec_3d_pos = P
        gaze_vec_3d_axis = (P-eyeball_3d_pos)*5

        gaze_pt_3d_0 = gaze_vec_3d_pos
        gaze_pt_3d_1 = gaze_vec_3d_pos + gaze_vec_3d_axis

        # calculate screen intersection
        intersections.append(geom_utils.ray_screen_intersect(eyeball_3d_pos, gaze_vec_3d_axis))

    if len(intersections) > 0:
        gaze_pt_mm = np.mean(intersections, axis=0)

        screen_vis = np.zeros((100, 160, 3))
        mm_to_px = np.array([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]]) * 1920.0/1200.0

        gaze_pt_px = mm_to_px.dot(gaze_pt_mm) + np.array([1920.0/2.0, 0, 0])

        return gaze_pt_px[:2]

    else: return np.array([-1000,-1000])


def mean_error(offset_vector):

    errors = []
    for fn in pkl_fns:
        pkl_file = open(os.path.join(path, fn))
        pt, face, eye0, eye1 = pickle.load(pkl_file)

        x = get_target_gaze_pos(pt)
        y = calc_gaze_pos(face, eye0, eye1, offset_vector)

        errors.append(np.linalg.norm(x-y))

    return np.mean(errors)

pkl_fns = [f for f in os.listdir(path) if f.endswith('.pkl')]

pkl_fns = [fn for fn in pkl_fns if int(pickle.load(open(os.path.join(path, fn)))[0]) < 7]

print scipy.optimize.fmin_bfgs(mean_error, [0.0, 0.0, 0.0, 1.0])