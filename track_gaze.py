import zmq
import cv2
import numpy as np
import re
import zmq_utils
import geom_utils

socket_img, socket_pts = zmq_utils.zmq_init()

fx, fy = 757.186370667, 757.260080183
cx, cy = 412.671083627, 272.671560372
camera_mat = np.array([[fx, 0,  cx],
                       [0,  fy, cy],
                       [0,  0,  0]], dtype=float)

socket_img, socket_pts = zmq_utils.zmq_init()


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

    intersections = []
    for i in range(2):

        iris_pts = eye0.iris_pts_3d if i == 0 else eye1.iris_pts_3d
        pupil = np.mean(iris_pts, axis=0)
        ray_dir = pupil / float(np.linalg.norm(pupil))

        # position eyeball in 3d
        eyeball_offset = pose_transform.dot(np.array([0, -2, 0, 1]))
        eyeball_3d_pos = (np.array(face.pts_3d[36+i*6])+np.array(face.pts_3d[39+i*6]))/2.0 + eyeball_offset[:3]

        try:
            P = geom_utils.ray_sphere_intersect((0,0,0), ray_dir, eyeball_3d_pos, 12)
        except geom_utils.NoIntersection:
            continue

        gaze_vec_3d_pos = P
        gaze_vec_3d_axis = (P-eyeball_3d_pos)*5

        gaze_pt_3d_0 = gaze_vec_3d_pos
        gaze_pt_3d_1 = gaze_vec_3d_pos + gaze_vec_3d_axis

        pts, _ = cv2.projectPoints(np.array([gaze_pt_3d_0, gaze_pt_3d_1]),
                                   np.eye(3, dtype=float),
                                   np.array([0, 0, 0], dtype=float), camera_mat, None)

        cv2.line(frame,
                 tuple(np.array(pts[0], int).squeeze()),
                 tuple(np.array(pts[1], int).squeeze()), (0,0,255))

        # calculate screen intersection
        intersections.append(geom_utils.ray_screen_intersect(eyeball_3d_pos, gaze_vec_3d_axis))

    if len(intersections) > 0:
        gaze_pt_mm = np.mean(intersections, axis=0)

        screen_vis = np.zeros((100, 160, 3))
        mm_to_px = np.array([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]]) * 160.0/100.0

        gaze_pt_px = mm_to_px.dot(gaze_pt_mm) + np.array([160/2.0, 0, 0])

        print gaze_pt_px

        cv2.circle(screen_vis, tuple(gaze_pt_px[:2].astype(int)), 4, [255]*3, -1)

        frame[10:10+100, 10:10+160] = screen_vis

    cv2.imshow("Frame (Python Client)", frame)


    key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     cv2.destroyAllWindows()
    #     video.release()
    #     break


