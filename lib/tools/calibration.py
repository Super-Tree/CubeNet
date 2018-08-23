
from __future__ import print_function

import numpy as np
import os
def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

    '''

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # =========================== 
    # ------- 2d to 3d ---------- 
    # =========================== 
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)
    def bbox2conner(self,box, rot):
        # box : x,y,z,l,w,h,..rot..
        vertices = np.zeros([8, 3], dtype=np.float32)

        # vertices[0] = np.array([0 - float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])
        # vertices[1] = np.array([0 - float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])
        # vertices[2] = np.array([0 + float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])
        # vertices[3] = np.array([0 + float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])
        #
        # vertices[4] = np.array([0 - float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])
        # vertices[5] = np.array([0 - float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])
        # vertices[6] = np.array([0 + float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])
        # vertices[7] = np.array([0 + float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])
        l = box[4]
        w = box[3]
        h = box[5]
        vertices = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                            [0, 0, 0, 0, h, h, h, h]])

        vertices = self.box_rot_trans(vertices.T, -rot, [box[0], box[1], box[2]])

        # return vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5], vertices[6], vertices[7],
        return vertices
    def box_rot_trans(self,vertices, rotation, translation):
        # points: numpy array;translation: moving scalar which should be small
        R = np.array([[np.cos(rotation), -np.sin(rotation), 0.],
                      [np.sin(rotation), np.cos(rotation), 0.],
                      [0, 0, 1]], dtype=np.float32)
        translation = np.reshape(translation, [3, 1])
        points_rot = np.add(np.matmul(R, vertices.transpose()), translation)

        return points_rot.transpose()
    def corners_to_img(self,corners):
        # TODO:hxd:declaration:more details in 'Vision meets Robotics:The KITTI Dataset'
        Tr=self.V2C
        R0=self.R0
        P2=self.P

        corners = corners.reshape(-1, 3)
        R0 = np.vstack((R0, np.zeros(3)))
        R0 = np.hstack((R0, np.zeros((4, 1))))
        R0[3, 3] = 1.0
        # print corners.shape

        Tr = Tr.reshape((3, 4))
        Tr = np.vstack((Tr, np.zeros((1, 4))))
        Tr[3, 3] = 1.0
        R0 = R0.reshape((4, 4))
        P2 = P2.reshape((3, 4))

        corners = np.vstack((corners.transpose(), np.ones(corners.shape[0])))

        # print(corners.shape)
        img_cor = reduce(np.dot, [P2, R0, Tr, corners])

        xs = img_cor[0, :] / np.abs(img_cor[2, :])
        ys = img_cor[1, :] / np.abs(img_cor[2, :])

        xmin = np.max([0, np.min(xs, axis=0)])  # in case minus coord
        xmax = np.max([0, np.max(xs, axis=0)])
        ymin = np.max([0, np.min(ys, axis=0)])
        ymax = np.max([0, np.max(ys, axis=0)])

        img_boxes = np.hstack((xmin, ymin, xmax, ymax))

        return np.around(img_boxes, 2)

class HangzhouCalib(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

    '''

    def __init__(self, calib_file_dir):

        calibs = self.load_calib(calib_file_dir)

        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P_rect_101']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from rectified camera  to Velodyne coord
        self.C2V = calibs['Tr_cam_to_velo']
        self.C2V = np.reshape(self.C2V, [3, 4])
        self.V2C = inverse_rigid_trans(self.C2V)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R_rect_101']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def load_calib(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        cam2velo = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_velo.txt'))
        Tr_cam_to_velo = np.zeros((3, 4))
        Tr_cam_to_velo[0:3, 0:3] = np.reshape(cam2velo['R'], [3, 3])
        Tr_cam_to_velo[:, 3] = cam2velo['T']
        data['Tr_cam_to_velo'] = np.reshape(Tr_cam_to_velo, [12])
        data['R_rect_101'] = cam2cam['R_rect_101']
        data['P_rect_101'] = cam2cam['P_rect_101']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        """pts_3d_velo:nx4:x y z i"""
        pts_3d_velo_in=pts_3d_velo[:,0:3]
        intensity=pts_3d_velo[:,3:4]

        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo_in)
        pts_3d_velo_rect = self.project_ref_to_rect(pts_3d_ref)

        return np.hstack((pts_3d_velo_rect,intensity))

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def velo3dpts_to_img2dpts(self,velo_pts):
        # TODO:hxd:declaration:more details in 'Vision meets Robotics:The KITTI Dataset'
        Tr=self.V2C
        R0=self.R0
        P2=self.P

        corners = velo_pts.reshape(-1, 3)
        R0 = np.vstack((R0, np.zeros(3)))
        R0 = np.hstack((R0, np.zeros((4, 1))))
        R0[3, 3] = 1.0
        # print corners.shape

        Tr = Tr.reshape((3, 4))
        Tr = np.vstack((Tr, np.zeros((1, 4))))
        Tr[3, 3] = 1.0
        R0 = R0.reshape((4, 4))
        P2 = P2.reshape((3, 4))

        corners = np.vstack((corners.transpose(), np.ones(corners.shape[0])))

        # print(corners.shape)
        img_cor = reduce(np.dot, [P2, R0, Tr, corners])

        xs = img_cor[0, :] / np.abs(img_cor[2, :])
        ys = img_cor[1, :] / np.abs(img_cor[2, :])

        img_boxes = np.stack((xs,ys)).T

        return np.around(img_boxes, 2)

    def corners_to_img(self,corners):
        # TODO:hxd:declaration:more details in 'Vision meets Robotics:The KITTI Dataset'
        Tr=self.V2C
        R0=self.R0
        P2=self.P

        corners = corners.reshape(-1, 3)
        R0 = np.vstack((R0, np.zeros(3)))
        R0 = np.hstack((R0, np.zeros((4, 1))))
        R0[3, 3] = 1.0
        # print corners.shape

        Tr = Tr.reshape((3, 4))
        Tr = np.vstack((Tr, np.zeros((1, 4))))
        Tr[3, 3] = 1.0
        R0 = R0.reshape((4, 4))
        P2 = P2.reshape((3, 4))

        corners = np.vstack((corners.transpose(), np.ones(corners.shape[0])))

        # print(corners.shape)
        img_cor = reduce(np.dot, [P2, R0, Tr, corners])

        xs = img_cor[0, :] / np.abs(img_cor[2, :])
        ys = img_cor[1, :] / np.abs(img_cor[2, :])

        xmin = np.max([0, np.min(xs, axis=0)])  # in case minus coord
        xmax = np.max([0, np.max(xs, axis=0)])
        ymin = np.max([0, np.min(ys, axis=0)])
        ymax = np.max([0, np.max(ys, axis=0)])

        img_boxes = np.hstack((xmin, ymin, xmax, ymax))

        return np.around(img_boxes, 2)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)
    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def rotx(self,points,t):
        ''' 3D Rotation about the x-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        R= np.array([[1,  0,  0,  0],
                     [0,  c, -s,  0],
                     [0,  s,  c,  0],
                     [0,  0,  0,1.0]])

        return np.dot(R,points.T).T

    def roty(self,points,t):
        ''' Rotation about the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)

        R= np.array([[c,  0,  s,  0],
                     [0,  1,  0,  0],
                     [-s, 0,  c,  0],
                     [0,  0,  0,1.0]])

        return np.dot(R,points.T).T

    def rotz(self,points,t):
        ''' Rotation about the z-axis. '''
        if len(points.shape)==2:
            assert points.shape[-1]==4,print('Input points shape is not [-1,4]')
        else:
            assert False, print('Input points shape is not [-1,4]')
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[c, -s, 0,  0],
                      [s, c,  0,  0],
                      [0, 0,  1,  0],
                      [0, 0, 0, 1.0]])

        return np.dot(R, points.T).T

    # ===========================
    # ------- box to conners ----------
    # ===========================
    def bbox2conner(self,box, rot):
        # box : x,y,z,l,w,h,..rot..
        vertices = np.zeros([8, 3], dtype=np.float32)

        # vertices[0] = np.array([0 - float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])
        # vertices[1] = np.array([0 - float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])
        # vertices[2] = np.array([0 + float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])
        # vertices[3] = np.array([0 + float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])
        #
        # vertices[4] = np.array([0 - float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])
        # vertices[5] = np.array([0 - float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])
        # vertices[6] = np.array([0 + float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])
        # vertices[7] = np.array([0 + float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])
        l = box[4]
        w = box[3]
        h = box[5]
        vertices = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                            [0, 0, 0, 0, h, h, h, h]])

        vertices = self.box_rot_trans(vertices.T, -rot, [box[0], box[1], box[2]])

        # return vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5], vertices[6], vertices[7],
        return vertices

    def box_rot_trans(self,vertices, rotation, translation):
        # points: numpy array;translation: moving scalar which should be small
        R = np.array([[np.cos(rotation), -np.sin(rotation), 0.],
                      [np.sin(rotation), np.cos(rotation), 0.],
                      [0, 0, 1]], dtype=np.float32)
        translation = np.reshape(translation, [3, 1])
        points_rot = np.add(np.matmul(R, vertices.transpose()), translation)

        return points_rot.transpose()

# def roty(t):
#     ''' Rotation about the y-axis. '''
#     c = np.cos(t)
#     s = np.sin(t)
#     return np.array([[c,  0,  s],
#                      [0,  1,  0],
#                      [-s, 0,  c]])
#
# def compute_box_3d(obj, P):
#     ''' Takes an object and a projection matrix (P) and projects the 3d
#         bounding box into the image plane.
#         Returns:
#             corners_2d: (8,2) array in left image coord.
#             corners_3d: (8,3) array in in rect camera coord.
#     '''
#     # compute rotational matrix around yaw axis
#     R = roty(obj.ry)
#
#     # 3d bounding box dimensions
#     l = obj.l
#     w = obj.w
#     h = obj.h
#
#     # 3d bounding box corners
#     x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
#     y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
#     z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
#
#     # rotate and translate 3d bounding box
#     corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
#     # print corners_3d.shape
#     corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
#     corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
#     corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
#     # print 'cornsers_3d: ', corners_3d
#     # only draw 3d bounding box for objs in front of the camera
#     # if np.any(corners_3d[2,:]<0.1):
#     #     corners_2d = None
#     #     return corners_2d, np.transpose(corners_3d)
#
#     # project the 3d bounding box into the image plane
#
#     # print 'corners_2d: ', corners_2d
#     return np.transpose(corners_3d)