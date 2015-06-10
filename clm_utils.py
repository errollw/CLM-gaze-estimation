from transformations import translation_matrix, euler_matrix

class FaceModel:

    def __init__(self, pts_3d_param, pts_2d_param, pose_param):

        self.pts_3d = pts_3d_param
        self.pts_2d = pts_2d_param
        self.pose = pose_param

    def get_rotation_transform(self):

        ex, ey, ez = self.pose[3:]
        return euler_matrix(ex, ey, ez)


class EyeModel:

    def __init__(self, pts_3d_param, pts_2d_param):

        self.pts_3d = pts_3d_param
        self.pts_2d = pts_2d_param

        self.iris_pts_2d = self.pts_2d[:8]
        self.iris_pts_3d = self.pts_3d[:8]
        self.lids_pts_2d = self.pts_2d[8:20]
        self.lids_pts_3d = self.pts_3d[8:20]

        # iris_pts = pts[:8]
        # lids_pts = pts[8:20]
        # pupil_pts = pts[20:28]