import os
import cv2
import pickle
import numpy as np

dir = "ground_truth_3d"
img_fns = [f for f in os.listdir("ground_truth_3d") if f.endswith(".png")]
img_fn = img_fns[0]
pkl_fn = img_fn.replace("png","pkl")

img = cv2.imread(os.path.join(dir, img_fn))
data = pickle.load(open(os.path.join(dir, pkl_fn), "rb"))

cam_mat = np.array([[749.9999,   0.0000,   400.0000],
                    [0.0000,   749.9999,   300.0000],
                    [0.0000,     0.0000,   1.0000]])

cv2.polylines(img, np.array([data["ldmks_lids_2d"]], int), True, 255)
cv2.polylines(img, np.array([data["ldmks_iris_2d"]], int), True, 255)

pts, _ = cv2.projectPoints(np.array([np.array(data["ldmks_iris_3d"])]),
                           np.eye(3, dtype=float),
                           np.array([0, 0, 0], dtype=float), cam_mat, None)

for pt in pts.squeeze().astype(int):
    x, y = pt
    cv2.circle(img, (800-x,y),2,255)

cv2.imshow("TEST", img)
cv2.waitKey(0)