
def parse_3d_pts(data_string):
    return [[float(x.translate(None, '[]')) for x in s.split(', ')] for s in data_string.split(';')]


def parse_2d_pts(data_string):
    pts_2d = [float(x.translate(None, '[]')) for x in data_string.split(',')]
    return zip(pts_2d[:len(pts_2d)/2], pts_2d[len(pts_2d)/2:])


def parse_floats(data_string):
    return [float(x.translate(None, '[]')) for x in data_string.split(',')]


def parse_pts_msg(data_string, center=False):

    datas = data_string.split('][')

    pose = parse_floats(datas[0])
    face_pts_3d = parse_3d_pts(datas[1])
    face_pts_2d = parse_2d_pts(datas[2])
    eye0_pts_3d = parse_3d_pts(datas[5])
    eye0_pts_2d = parse_2d_pts(datas[6])
    eye1_pts_3d = parse_3d_pts(datas[3])
    eye1_pts_2d = parse_2d_pts(datas[4])

    return pose, face_pts_3d, face_pts_2d,\
           eye0_pts_3d, eye0_pts_2d, eye1_pts_3d, eye1_pts_2d