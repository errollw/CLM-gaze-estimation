import os
import cv2
import pickle
import numpy as np
import transformations
import scipy.optimize
import matplotlib.pyplot as plt
import visualize

from scipy.spatial.distance import cdist

def dehomo(vector):
    return vector[:-1]/vector[-1]

def point_in_poly(p,vs):

    for i in range(len(vs)):
        v0, v1 = np.array(vs[i]), np.array(vs[(i+1)%len(vs)])
        edge = np.add(v1, -v0)
        normal = np.array([[0, 1], [-1, 0]]).dot(edge)
        vec = np.add(np.array(p), -v0)
        if (normal.dot(vec) > 0): return False

    return True

dir = "ground_truth_3d"
img_fns = [f for f in os.listdir("ground_truth_3d") if f.endswith(".png")]
img_fn = img_fns[2]
pkl_fn = img_fn.replace("png", "pkl")
clm_pkl_fn = "clm_%s"%pkl_fn

img = cv2.imread(os.path.join(dir, img_fn))
data = pickle.load(open(os.path.join(dir, pkl_fn), "rb"))
face, eye0, eye1 = pickle.load(open(os.path.join(dir, clm_pkl_fn), "rb"))

offset = [0, -2, 10, 1]
eyeball_offset_clm = face.get_rotation_transform().dot(np.array(offset))
eyeball_3d_pos_clm = (np.array(face.pts_3d[42])+np.array(face.pts_3d[45]))/2.0 + eyeball_offset_clm[:3]

true_iris_centre_3d = np.mean(data["ldmks_iris_3d"], axis=0)*100

cam_mat = np.array([[749.9999,   0.0000,   400.0000],
                    [0.0000,   749.9999,   300.0000],
                    [0.0000,     0.0000,   1.0000]])

coord_swap = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

visualize.visualize(face.pts_3d)
visualize.visualize([coord_swap.dot(np.array(data["eye_centre_3d"])*100)])
visualize.visualize([eyeball_3d_pos_clm], radius=12)

# VISUALIZE IN 2D BEFORE OPTIMIZING

for pt in face.pts_2d:
    cv2.circle(img, tuple([int(p) for p in pt]), 3, 0, -1)
    cv2.circle(img, tuple([int(p) for p in pt]), 2, (255, 255, 255), -1)
cv2.polylines(img, np.array([data["ldmks_lids_2d"]], int), True, 255)
cv2.imshow("TEST", img)
cv2.waitKey(1)

#

(e_x, e_y), (e_w, e_h), e_tht = cv2.fitEllipse(np.array(data["ldmks_iris_2d"], dtype=int))
pts_iris = cv2.ellipse2Poly((int(e_x), int(e_y)),
                            (int(e_w/2.0), int(e_h/2.0)),
                            int(e_tht), 0, 360, 5)

visible_pts = [pt for pt in pts_iris if point_in_poly(pt, data["ldmks_lids_2d"])]
for pt in visible_pts:
    cv2.circle(img, tuple(pt), 2, (255, 255, 255), -1)

eye_centre = np.array(data["eye_centre_3d"])*100 # [ -42.22064747  -53.62063667  443.24687103]
# eye_centre = coord_swap.dot([ -42.02786715,-49.4106491, 423.94131185])
# eye_centre = coord_swap.dot(eyeball_3d_pos_clm)

pt = cam_mat.dot(coord_swap.dot(np.array(data["eye_centre_3d"])*100))
x, y = dehomo(pt).astype(int)
cv2.circle(img, (x,y), 2, (255, 0, 255), -1)

pt = cam_mat.dot(coord_swap.dot(eye_centre))
x, y = dehomo(pt).astype(int)
cv2.circle(img, (x,y), 2, (0, 255, 255), -1)

accuracy = 50

def calc_error(args, debug=False):

    # angle_x, angle_y, o_x, o_y, o_z = args
    angle_x, angle_y = args

    x = np.array([1, 0, 0, 1])
    R_x = transformations.rotation_matrix(angle_x, [1, 0, 0], eye_centre)
    R_y = transformations.rotation_matrix(angle_y, [0, 1, 0], eye_centre)
    S = transformations.scale_matrix(6)
    T = transformations.translation_matrix(eye_centre)
    # T = transformations.translation_matrix(eye_centre + np.array([o_x*100, o_y*100, o_z*100]))
    T2 = transformations.translation_matrix([0,0,-12])

    if debug:
        trans = transformations.concatenate_matrices(*reversed([T, T2, R_x, R_y]))
        pt = dehomo(trans.dot([0,0,-50,1]))
        cv2.line(img,
                 tuple(dehomo(cam_mat.dot(coord_swap.dot(true_iris_centre_3d))).astype(int)),
                 tuple(dehomo(cam_mat.dot(coord_swap.dot(pt))).astype(int)),
                 (255, 255, 0))

    est_pts = []
    for t in np.linspace(0, np.pi*2, accuracy):

        R = transformations.rotation_matrix(t, [0, 0, 1])
        trans = transformations.concatenate_matrices(*reversed([R, S, T, T2, R_x, R_y]))

        threeD_pt = coord_swap.dot(dehomo(trans.dot(x)))

        pt = cam_mat.dot(threeD_pt)

        if point_in_poly(dehomo(pt).astype(int), data["ldmks_lids_2d"]):
            est_pts.append(dehomo(pt).astype(int))
            if debug: cv2.circle(img, tuple(dehomo(pt).astype(int)), 1, (255, 0, 0), -1)

    try:
        D = cdist(est_pts, visible_pts, 'euclidean')
        H1 = np.max(np.min(D, axis=1))
        H2 = np.max(np.min(D, axis=0))
        return (H1 + H2) / 2.0
    except ValueError:
        return 20

x, f, d = scipy.optimize.fmin_l_bfgs_b(calc_error, [0.0, 0.0], approx_grad=True, epsilon=0.01)

print x, f, d

calc_error(x, debug=True)

angle_x, angle_y, o_x, o_y, o_z = x
pt = cam_mat.dot(np.array([o_x, o_y, o_z]))
x, y = dehomo(pt).astype(int)
cv2.circle(img, (x,y), 2, (0, 255, 255), -1)

gaze_dir = true_iris_centre_3d-np.array(data["eye_centre_3d"])*100
gaze_dir /= np.linalg.norm(gaze_dir)
cv2.line(img,
         tuple(dehomo(cam_mat.dot(coord_swap.dot(true_iris_centre_3d))).astype(int)),
         tuple(dehomo(cam_mat.dot(coord_swap.dot(true_iris_centre_3d+gaze_dir*50))).astype(int)),
         (255,0,255))

cv2.imshow("TEST", img)
cv2.waitKey(0)