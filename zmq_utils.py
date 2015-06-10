import zmq
from clm_utils import FaceModel, EyeModel


def parse_3d_pts(data_string):
    return [[float(x.translate(None, '[]')) for x in s.split(', ')] for s in data_string.split(';')]


def parse_2d_pts(data_string):
    pts_2d = [float(x.translate(None, '[]')) for x in data_string.split(',')]
    return zip(pts_2d[:len(pts_2d)/2], pts_2d[len(pts_2d)/2:])


def parse_floats(data_string):
    return [float(x.translate(None, '[]')) for x in data_string.split(',')]


def parse_pts_msg(data_string, center=False):

    ds = data_string.split('][')

    face = FaceModel(parse_3d_pts(ds[1]), parse_2d_pts(ds[2]), parse_floats(ds[0]))
    eye0 = EyeModel(parse_3d_pts(ds[5]), parse_2d_pts(ds[6]))
    eye1 = EyeModel(parse_3d_pts(ds[3]), parse_2d_pts(ds[4]))

    return face, eye0, eye1


def zmq_init():

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

    return socket_img, socket_pts