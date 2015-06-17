import os
import cv2
import pickle
import numpy as np
import transformations
import scipy.optimize
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

dir = "ground_truth_3d"
img_fns = [f for f in os.listdir("ground_truth_3d") if f.endswith(".png")]
img_fn = img_fns[0]
pkl_fn = img_fn.replace("png", "pkl")

img = cv2.imread(os.path.join(dir, img_fn))
data = pickle.load(open(os.path.join(dir, pkl_fn), "rb"))

cam_mat = np.array([[749.9999,   0.0000,   400.0000],
                    [0.0000,   749.9999,   300.0000],
                    [0.0000,     0.0000,   1.0000]])

def dehomogenize(vector):
    return vector[:-1]/vector[-1]


def point_in_poly(p,vs):

    for i in range(len(vs)):
        v0, v1 = np.array(vs[i]), np.array(vs[(i+1)%len(vs)])
        edge = np.add(v1, -v0)
        normal = np.array([[0, 1], [-1, 0]]).dot(edge)
        vec = np.add(np.array(p), -v0)
        if (normal.dot(vec) > 0): return False

    return True


cv2.polylines(img, np.array([data["ldmks_lids_2d"]], int), True, 255)
# cv2.polylines(img, np.array([data["ldmks_iris_2d"]], int), True, 255)

(e_x, e_y), (e_w, e_h), e_tht = cv2.fitEllipse(np.array(data["ldmks_iris_2d"], dtype=int))
pts_iris = cv2.ellipse2Poly((int(e_x), int(e_y)),
                            (int(e_w/2.0), int(e_h/2.0)),
                            int(e_tht), 0, 360, 5)

visible_pts = [pt for pt in pts_iris if point_in_poly(pt, data["ldmks_lids_2d"])]
for pt in visible_pts:
    cv2.circle(img, tuple(pt), 1, (50, 0, 0), -1)

print data["eye_centre_3d"]


coord_swap = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

pt = cam_mat.dot(coord_swap.dot(data["eye_centre_3d"]))
x, y = dehomogenize(pt).astype(int)
cv2.circle(img, (x,y), 2, (255, 255, 255), -1)

def calc_error(args):

    angle = args[0]

    est_pts = []
    for t in np.linspace(0, np.pi*2, 50):
        x = np.array([1, 0, 0, 1])
        R = transformations.rotation_matrix(t, [0, 0, 1])
        R2 = transformations.rotation_matrix(angle, [0, 1, 0], data["eye_centre_3d"])
        S = transformations.scale_matrix(0.06)
        T = transformations.translation_matrix(data["eye_centre_3d"])
        T2 = transformations.translation_matrix([0,0,-0.12])

        trans = transformations.concatenate_matrices(*reversed([R,S,T,T2,R2]))

        threeD_pt = coord_swap.dot(dehomogenize(trans.dot(x)))

        pt = cam_mat.dot(threeD_pt)
        x, y = dehomogenize(pt).astype(int)

        if point_in_poly(dehomogenize(pt).astype(int), data["ldmks_lids_2d"]):
            est_pts.append(dehomogenize(pt).astype(int))
            cv2.circle(img, tuple(dehomogenize(pt).astype(int)), 1, (255, 0, 0), -1)

    try:
        D = cdist(est_pts, visible_pts, 'euclidean')
        H1 = np.max(np.min(D, axis=1))
        H2 = np.max(np.min(D, axis=0))
        return (H1 + H2) / 2.0
    except ValueError:
        return 20

xs = np.linspace(0, np.pi*2, 100)
ys = [calc_error([x]) for x in xs]

plt.plot(xs, ys)
plt.show()

# print calc_error(np.pi/32.0)

print scipy.optimize.fmin_l_bfgs_b(calc_error, [0.0], approx_grad=True, epsilon=0.1)

# calc_error([0.03784254])


cv2.imshow("TEST", img)
cv2.waitKey(0)