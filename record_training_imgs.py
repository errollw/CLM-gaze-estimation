import zmq
import cv2
import numpy as np
import zmq_utils
import geom_utils
import pickle
import os
from time import time

fx, fy = 757.186370667, 757.260080183
cx, cy = 412.671083627, 272.671560372
camera_mat = np.array([[fx, 0,  cx],
                       [0,  fy, cy],
                       [0,  0,  0]], dtype=float)

socket_img, socket_pts = zmq_utils.zmq_init()

# display calibration pattern
calibration_pattern = np.ones((1200, 1920), dtype='uint8')*128
for i in range(3):
    for j in range(3):
        pos = (1720/2 * j + 100, 1000/2 * i + 100)
        text_pos = (pos[0]-10, pos[1]+10)
        cv2.circle(calibration_pattern, pos, 30, 255, -1)
        cv2.putText(calibration_pattern, str(i*3+j+1), text_pos, cv2.FONT_HERSHEY_PLAIN, 2.0, 0)

cv2.imshow("Calibration Pattern", calibration_pattern)

path = "gaze_calibration_imgs"


while True:

    data_img = socket_img.recv()
    data_pts = socket_pts.recv()

    # parse image data
    frame = np.fromstring(data_img, dtype='uint8')
    frame = frame.reshape((600, 800, 3))

    # parse scene data
    face, eye0, eye1 = zmq_utils.parse_pts_msg(data_pts)
    pose_transform = face.get_rotation_transform()

    # draw 2d face pts on img
    for pt in face.pts_2d[:36]:
        cv2.circle(frame, tuple([int(p) for p in pt]), 3, 0, -1)
        cv2.circle(frame, tuple([int(p) for p in pt]), 2, (255, 255, 255), -1)

    # draw lines about the eyes
    cv2.polylines(frame, np.array([eye0.iris_pts_2d], dtype=int), True, (0, 0, 255))
    cv2.polylines(frame, np.array([eye1.iris_pts_2d], dtype=int), True, (0, 0, 255))
    cv2.polylines(frame, np.array([eye0.lids_pts_2d], dtype=int), True, 255)
    cv2.polylines(frame, np.array([eye1.lids_pts_2d], dtype=int), True, 255)

    cv2.imshow("Frame (Python Client)", frame)

    key = cv2.waitKey(1) & 0xFF

    if chr(key) == 'q':
        cv2.destroyAllWindows()
        break

    if chr(key) in '12345678':
        cv2.imwrite(os.path.join(path, "%d.png"%time()), frame)
        plk_file = open(os.path.join(path, "%d.pkl"%time()), "wb")
        pickle.dump((chr(key), face, eye0, eye1), plk_file)

