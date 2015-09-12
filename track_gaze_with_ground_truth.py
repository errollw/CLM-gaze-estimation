import zmq
import cv2
import os
import pickle
import numpy as np
import zmq_utils
import geom_utils
import visualize
from time import time

cam_mat = np.array([[749.9999,   0.0000,   400.0000],
                    [0.0000,   749.9999,   300.0000],
                    [0.0000,     0.0000,   1.0000]])

socket_img, socket_pts = zmq_utils.zmq_init()

f_idx = 0

def dehomo(vector):
    return vector[:-1]/vector[-1]

dir, fn = "ground_truth_3d", "1434636738"
img = cv2.imread(os.path.join(dir, "%s.png"%fn))
truth_data = pickle.load(open(os.path.join(dir, "%s.pkl"%fn), "rb"))

coord_swap = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

true_iris_centre_3d = np.mean(truth_data["ldmks_iris_3d"], axis=0)*100
gaze_dir = true_iris_centre_3d-np.array(truth_data["eye_centre_3d"])*100
gaze_dir /= np.linalg.norm(gaze_dir)

recording = False

while True:

    data_img = socket_img.recv()
    data_pts = socket_pts.recv()

    # parse image data
    frame = np.fromstring(data_img, dtype='uint8')
    frame = frame.reshape((600, 800, 3))
    frame_copy = frame.copy()

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

    intersections = []
    for i in range(2):

        iris_pts = eye0.iris_pts_3d if i == 0 else eye1.iris_pts_3d
        pupil = np.mean(iris_pts, axis=0)
        ray_dir = pupil / float(np.linalg.norm(pupil))

        # position eyeball in 3d
        #[ 1.26128993 -2.2593072   3.70121942  6.11718562]
        offset = [0, 0, 0, 1]
        if i == 1: offset = np.array([-1, 1, 1, 1]) * offset
        eyeball_offset = pose_transform.dot(np.array(offset))
        eyeball_3d_pos = (np.array(face.pts_3d[36+i*6])+np.array(face.pts_3d[39+i*6]))/2.0 + eyeball_offset[:3]

        try:
            P = geom_utils.ray_sphere_intersect((0,0,0), ray_dir, eyeball_3d_pos, 6.11)
        except geom_utils.NoIntersection:
            continue

        gaze_vec_3d_pos = P
        gaze_vec_3d_axis = (P-eyeball_3d_pos)*5

        gaze_pt_3d_0 = gaze_vec_3d_pos
        gaze_pt_3d_1 = gaze_vec_3d_pos + gaze_vec_3d_axis

        pts, _ = cv2.projectPoints(np.array([gaze_pt_3d_0, gaze_pt_3d_1]),
                                   np.eye(3, dtype=float),
                                   np.array([0, 0, 0], dtype=float), cam_mat, None)

        cv2.line(frame,
                 tuple(np.array(pts[0], int).squeeze()),
                 tuple(np.array(pts[1], int).squeeze()), (0,0,255))

        if i==1:
            print coord_swap.dot(np.array(truth_data["eye_centre_3d"])*100), eyeball_3d_pos

            pt = cam_mat.dot(coord_swap.dot(truth_data["eye_centre_3d"])*100)
            cv2.circle(frame, tuple(dehomo(pt).astype(int)), 2, (255, 255, 255))

            pt = cam_mat.dot(eyeball_3d_pos)
            cv2.circle(frame, tuple(dehomo(pt).astype(int)), 2, (128, 128, 128), -1)

            cv2.line(frame,
                     tuple(dehomo(cam_mat.dot(coord_swap.dot(true_iris_centre_3d))).astype(int)),
                     tuple(dehomo(cam_mat.dot(coord_swap.dot(true_iris_centre_3d+gaze_dir*50))).astype(int)),
                     (0,255,255))

        # calculate screen intersection
        intersections.append(geom_utils.ray_screen_intersect(eyeball_3d_pos, gaze_vec_3d_axis))

    if len(intersections) > 0:
        gaze_pt_mm = np.mean(intersections, axis=0)

        screen_vis = np.zeros((100, 160, 3))
        mm_to_px = np.array([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]]) * 160.0/100.0

        gaze_pt_px = mm_to_px.dot(gaze_pt_mm) + np.array([160/2.0, 0, 0])

        cv2.circle(screen_vis, tuple(gaze_pt_px[:2].astype(int)), 40, [128]*3, -1)
        cv2.circle(screen_vis, tuple(gaze_pt_px[:2].astype(int)), 4, [255]*3, -1)

        frame[10:10+100, 10:10+160] = screen_vis

    cv2.imshow("Frame (Python Client)", frame)

    if recording: cv2.imwrite("vid_imgs/%d.jpg"%f_idx, frame)
    f_idx += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    if key == ord('s'):
        pickle.dump([face, eye0, eye1], open(os.path.join(dir, "clm_%s.pkl"%fn), "wb"))

