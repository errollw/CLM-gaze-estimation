import numpy as np


cam_mat = np.array([[749.9999,   0.0000,   400.0000],
                    [0.0000,   749.9999,   300.0000],
                    [0.0000,     0.0000,   1.0000]])


coord_swap = np.array([[1,  0,  0],
                       [0, -1,  0],
                       [0,  0, -1]])


def point_in_poly(p,vs):

    for i in range(len(vs)):
        v0, v1 = np.array(vs[i]), np.array(vs[(i+1)%len(vs)])
        edge = np.add(v1, -v0)
        normal = np.array([[0, 1], [-1, 0]]).dot(edge)
        vec = np.add(np.array(p), -v0)
        if normal.dot(vec) > 0: return False

    return True


def dehomo(vector):

    return vector[:-1]/vector[-1]


def pitch_yaw_to_vec((p,y)):

    x = np.cos(y)*np.cos(p)
    y = np.sin(y)*np.cos(p)
    z = np.sin(p)
    return (x,y,z)