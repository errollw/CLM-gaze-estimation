import sys
import zmq
import cv2
import numpy as np
import re
from pprint import pprint
from visual import *

# ZMQ setup
context = zmq.Context()

# socket to receive image
socket_img = context.socket(zmq.SUB)
socket_img.connect("tcp://localhost:5555")
socket_img.setsockopt_string(zmq.SUBSCRIBE, unicode(''))

# socket to receive data
socket_pts = context.socket(zmq.SUB)
socket_pts.connect("tcp://localhost:5556")
socket_pts.setsockopt_string(zmq.SUBSCRIBE, unicode(''))

# for swapping between OpenCV and VPython coordinates
coord_swap = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

fx, fy = 757.186370667, 757.260080183
cx, cy = 412.671083627, 272.671560372
camera_mat = np.array([[fx, 0,  cx],
                       [0,  fy, cy],
                       [0,  0,  0]], dtype=float)

# initialize landmark geometry for drawing
face_spheres_3d = []
for _ in range(68):
    face_spheres_3d.append(sphere(pos=(0, 0, 0), radius=1, color=color.red))

# init eyeball geometry
eyeballs_3d = [sphere(pos=(0, 0, 0), radius=12, color=color.white, opacity=0.5),
               sphere(pos=(0, 0, 0), radius=12, color=color.white, opacity=0.5)]
gaze_vecs_3d = [arrow(pos=(0,0,0), axis=(1,0,0), shaftwidth=0.5, color=color.blue),
                arrow(pos=(0,0,0), axis=(1,0,0), shaftwidth=0.5, color=color.blue)]

head_pose_vec_3d = arrow(pos=(0,0,0), axis=(5,0,0), shaftwidth=10, color=color.red)

def ray_sphere_intersect(ray_origin, ray_dir, sphere_origin, sphere_radius):

    dx, dy, dz = ray_dir
    x0, y0, z0 = ray_origin
    cx, cy, cz = sphere_origin
    R = sphere_radius

    a = dx*dx + dy*dy + dz*dz
    b = 2*dx*(x0-cx) + 2*dy*(y0-cy) + 2*dz*(z0-cz)
    c = cx*cx + cy*cy + cz*cz + x0*x0 + y0*y0 + z0*z0 + -2*(cx*x0 + cy*y0 + cz*z0) - R*R

    disc = b*b - 4*a*c

    t = (-b - sqrt(disc))/2*a

    return vector(ray_origin) + vector(ray_dir)*t


def ray_screen_intersect(ray_origin, ray_dir):

    dx, dy, dz = ray_dir
    x0, y0, z0 = ray_origin

    t = -z0/float(dz)

    return vector(ray_origin) + vector(ray_dir)*t


def screen_mm_to_px(gaze_pos_mm):

    dx, dy, dz = ray_dir

    return gaze_pos_mm + vector(-27, 4, 0) / 27.0 * 1080


def parse_3d_pts(data_string):
    return [[float(x.translate(None, '[]')) for x in s.split(', ')] for s in data_string.split(';')]


def parse_2d_pts(data_string):
    pts_2d = [float(x.translate(None, '[]')) for x in data_string.split(',')]
    return zip(pts_2d[:len(pts_2d)/2], pts_2d[len(pts_2d)/2:])


def parse_floats(data_string):
    return [float(x.translate(None, '[]')) for x in data_string.split(',')]


def parse_pts_msg(data_string, center=False):

    datas = data_string.split('][')

    params_global = parse_floats(datas[0])
    face_pts_3d = parse_3d_pts(datas[1])
    face_pts_2d = parse_2d_pts(datas[2])
    eye0_pts_3d = parse_3d_pts(datas[5])
    eye0_pts_2d = parse_2d_pts(datas[6])
    eye1_pts_3d = parse_3d_pts(datas[3])
    eye1_pts_2d = parse_2d_pts(datas[4])

    return params_global, face_pts_3d, face_pts_2d,\
           eye0_pts_3d, eye0_pts_2d, eye1_pts_3d, eye1_pts_2d


def split_up_eye_pts(pts):

    iris_pts = pts[:8]
    lids_pts = pts[8:20]
    pupil_pts = pts[20:28]

    return iris_pts, lids_pts, pupil_pts



while True:

    print 'waiting...'
    data_img = socket_img.recv()
    data_pts = socket_pts.recv()

    # parse image data
    frame = np.fromstring(data_img, dtype='uint8')
    frame = frame.reshape((600,800,3))

    # parse pts data
    params_global, face_pts_3d, face_pts_2d, \
    eye0_pts_3d, eye0_pts_2d, eye1_pts_3d, eye1_pts_2d = parse_pts_msg(data_pts)

    Tx, Ty, Tz, Eul_x, Eul_y, Eul_z = params_global

    rot_mat = cv2.Rodrigues(np.array([Eul_x, Eul_y, Eul_z]))[0]

    head_pose_vec_3d.pos = coord_swap.dot(np.array([Tx, Ty, Tz]))
    head_pose_vec_3d.axis = coord_swap.dot(rot_mat.dot(np.array([0,0,-1])))*50.0

    eye0_iris_pts_2d, eye0_lids_pts_2d, eye0_pupil_pts_2d = split_up_eye_pts(eye0_pts_2d)
    eye1_iris_pts_2d, eye1_lids_pts_2d, eye1_pupil_pts_2d = split_up_eye_pts(eye1_pts_2d)

    # draw 2d face pts on img
    for pt in face_pts_2d[:36]:
        cv2.circle(frame, tuple([int(p) for p in pt]), 3, 0, -1)
        cv2.circle(frame, tuple([int(p) for p in pt]), 2, (255,255,255), -1)

    # draw 2d eye pts
    # for pt in eye0_iris_pts_2d + eye1_iris_pts_2d:
    #     cv2.circle(frame, tuple([int(p) for p in pt]), 2, (0,0,255), -1)
    #
    # for pt in eye0_lids_pts_2d + eye1_lids_pts_2d:
    #     cv2.circle(frame, tuple([int(p) for p in pt]), 2, (255,0,0), -1)

    cv2.polylines(frame, np.array([eye0_iris_pts_2d],dtype=int), True, (0,0,255))
    cv2.polylines(frame, np.array([eye1_iris_pts_2d],dtype=int), True, (0,0,255))
    cv2.polylines(frame, np.array([eye0_lids_pts_2d],dtype=int), True, 255)
    cv2.polylines(frame, np.array([eye1_lids_pts_2d],dtype=int), True, 255)

    # position 3d face pts
    for i, pt in enumerate(face_pts_3d):
        face_spheres_3d[i].pos = coord_swap.dot(np.array(pt).T)

    intersections = []
    for i in range(2):

        iris_pts = eye0_pts_3d[:8] if i == 0 else eye1_pts_3d[:8]
        pupil = coord_swap.dot(np.mean(iris_pts, axis=0))
        ray_dir = pupil / float(np.linalg.norm(pupil))

        # position eyeball in 3d
        eyeball_offset = coord_swap.dot(rot_mat.dot(np.array([0,-2,0])))
        eyeballs_3d[i].pos = (face_spheres_3d[36+i*6].pos+face_spheres_3d[39+i*6].pos)/2.0 + eyeball_offset

        P = ray_sphere_intersect((0,0,0), ray_dir, eyeballs_3d[i].pos, 12)
        gaze_vecs_3d[i].pos = P
        gaze_vecs_3d[i].axis = (P-eyeballs_3d[i].pos)*5

        gaze_pt_3d_0 = coord_swap.dot(gaze_vecs_3d[i].pos)
        gaze_pt_3d_1 = coord_swap.dot(gaze_vecs_3d[i].pos + gaze_vecs_3d[i].axis)

        pts, _ = cv2.projectPoints(np.array([gaze_pt_3d_0, gaze_pt_3d_1]), np.eye(3, dtype=float), np.array([0,0,0], dtype=float), camera_mat, None)

        cv2.line(frame,
                 tuple(np.array(pts[0], int).squeeze()),
                 tuple(np.array(pts[1], int).squeeze()), (0,0,255))

        # calculate screen intersection
        intersections.append(ray_screen_intersect(eyeballs_3d[i].pos, norm(gaze_vecs_3d[i].axis)))

    cv2.imshow("Frame (Python Client)", frame)


    gaze_pt_mm = np.mean(intersections, axis=0)
    # print "%.2f,%.2f,%.2f"%tuple(gaze_pt_mm)

    # print screen_mm_to_px(np.mean(intersections, axis=0))

    mm_to_px = np.array([[-1,  0, 0],
                         [ 0, -1, 0],
                         [ 0,  0, 1]]) * 1920.0/270.0

    gaze_pt_px = mm_to_px.dot(gaze_pt_mm) + np.array([1920/2.0, 0, 0])

    print "%.2f,%.2f,%.2f"%tuple(gaze_pt_px)

    key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     cv2.destroyAllWindows()
    #     video.release()
    #     break


