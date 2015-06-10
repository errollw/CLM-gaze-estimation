import numpy as np
from math import sqrt


class NoIntersection(Exception):
    def __init__(self, msg):
        self.msg = msg


def ray_sphere_intersect(ray_origin, ray_dir, sphere_origin, sphere_radius):

    dx, dy, dz = ray_dir
    x0, y0, z0 = ray_origin
    cx, cy, cz = sphere_origin
    r = sphere_radius

    a = dx*dx + dy*dy + dz*dz
    b = 2*dx*(x0-cx) + 2*dy*(y0-cy) + 2*dz*(z0-cz)
    c = cx*cx + cy*cy + cz*cz + x0*x0 + y0*y0 + z0*z0 + -2*(cx*x0 + cy*y0 + cz*z0) - r*r

    disc = b*b - 4*a*c

    if disc < 0:
        raise NoIntersection('Negative discriminant')

    t = (-b - sqrt(b*b - 4*a*c))/2*a

    return np.array(ray_origin) + np.array(ray_dir)*t


def ray_screen_intersect(ray_origin, ray_dir):

    dx, dy, dz = ray_dir
    x0, y0, z0 = ray_origin

    t = -z0/float(dz)

    return np.array(ray_origin) + np.array(ray_dir)*t