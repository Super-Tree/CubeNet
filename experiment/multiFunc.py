import numpy as np
import cv2
# import rospy
# from sensor_msgs.msg import PointCloud
import os
import matplotlib.pyplot as plt
import random
import mayavi.mlab as mlab  # 3d point
import cPickle as pickle

from sensor_msgs.msg import PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# Add ROS  to PYTHONPATH
add_path("/opt/ros/indigo/lib/python2.7/dist-packages")


def point_cloud_to_panorama(points, v_res=0.42, h_res=0.20, v_fov=(-24.9, 2.0), d_range=(0, 100), y_fudge=3,
                            side_range=(-30., 30.), fwd_range=(0., 60), height_range=(-2, 0.4)):
    """ Takes point cloud data as input and creates a 360 degree panoramic
        image, returned as a numpy array.

    Args:
        points: (np array)
            The numpy array containing the point cloud. .
            The shape should be at least Nx3 (allowing for more columns)
            - Where N is the number of points, and
            - each point is specified by at least 3 values (x, y, z)
        v_res: (float)
            vertical angular resolution in degrees. This will influence the
            height of the output image.
        h_res: (float)
            horizontal angular resolution in degrees. This will influence
            the width of the output image.
        v_fov: (tuple of two floats)
            Field of view in degrees (-min_negative_angle, max_positive_angle)
        d_range: (tuple of two floats) (default = (0,100))
            Used for clipping distance values to be within a min and max range.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical image height do not match the actual data.
    Returns:
        A numpy array representing a 360 degree panoramic image of the point
        cloud.
    """
    # side_range = (-30., 30.)
    # fwd_range = (0., 60)
    # height_range = (-2, 0.4)  #
    xi_points = points[:, 0]
    yi_points = points[:, 1]
    zi_points = points[:, 2]
    reflectance = points[:, 3]

    f_filt = np.logical_and(
        (xi_points > fwd_range[0]), (xi_points < fwd_range[1]))
    s_filt = np.logical_and(
        (yi_points > -side_range[1]), (yi_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    z_filt = np.logical_and((zi_points >= height_range[0]),
                            (zi_points < height_range[1]))
    zfilter = np.logical_and(filter, z_filt)
    indices = np.argwhere(zfilter).flatten()
    print 'indice size'
    print indices.size

    x_points = xi_points[indices]
    print 'xi_points'
    print x_points
    y_points = yi_points[indices]
    z_points = zi_points[indices]
    r_points = reflectance[indices]
    r_max = max(r_points)
    z_max = max(z_points)
    r_min = min(r_points)
    z_min = min(z_points)

    # Projecting to 2D
    # x_points = points[:, 0]
    # y_points = points[:, 1]
    # z_points = points[:, 2]
    # r_points = points[:, 3]

    # d_points = np.sqrt(x_points ** 2 + y_points ** 2)  # map distance relative to origin
    # print 'd_points size', len(d_points)
    d_points = np.sqrt(x_points ** 2 + y_points ** 2 + z_points ** 2)  # abs distance
    # d_points = r_points
    # d_points = z_points

    # d_points = np.zeros(indices.size)
    # for i in range(indices.size):
    #     d_points[i] = z_points[i]

    # We use map distance, because otherwise it would not project onto a cylinder,
    # instead, it would map onto a segment of slice of a sphere.

    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_fov_total = -v_fov[0] + v_fov[1]

    # CONVERT TO RADIANS
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # MAPPING TO CYLINDER
    de_points = np.sqrt(x_points ** 2 + y_points ** 2)
    x_img = np.arctan2(y_points, x_points) / h_res_rad
    y_img = -(np.arctan2(z_points, de_points) / v_res_rad)

    # THEORETICAL MAX HEIGHT FOR IMAGE
    d_plane = (v_fov_total / v_res) / (v_fov_total * (np.pi / 180))
    h_below = d_plane * np.tan(-v_fov[0] * (np.pi / 180))
    h_above = d_plane * np.tan(v_fov[1] * (np.pi / 180))
    y_max = int(np.ceil(h_below + h_above + y_fudge))

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -180.0 / h_res / 2
    x_img = np.trunc(-x_img - x_min).astype(np.int32)
    x_max = int(np.ceil(180.0 / h_res))

    y_min = -((v_fov[1] / v_res) + y_fudge)
    y_img = np.trunc(y_img - y_min).astype(np.int32)

    # CLIP DISTANCES
    d_points = np.clip(d_points, a_min=d_range[0], a_max=d_range[1])

    # CONVERT TO IMAGE ARRAY
    img = np.zeros([y_max + 1, x_max + 1, 3], dtype=np.uint8)
    distance = np.sqrt(x_points ** 2 + y_points ** 2 + z_points ** 2)
    dis_max = max(distance)
    dis_min = min(distance)
    img[y_img, x_img, 0] = scale_to_255(distance, min=dis_min, max=dis_max)
    img[y_img, x_img, 1] = scale_to_255(z_points, min=z_min, max=z_max)
    img[y_img, x_img, 2] = scale_to_255(r_points, min=r_min, max=r_max)

    return img


def lidar_point_to_camera(pnt):
    """
    convert lidar point to camera
    """
    point = np.zeros((4, 1))
    point[0] = pnt[0]
    point[1] = pnt[1]
    point[2] = pnt[2]

    P = np.array(
        [[7.53374491e-03, -9.99971390e-01, -6.16602018e-04, -4.06976603e-03],
         [1.48024904e-02, 7.28073297e-04, -9.99890208e-01, -7.63161778e-02],
         [9.99862075e-01, 7.52379000e-03, 1.48075502e-02, -2.71780610e-01]]
    )
    R = P[:, :3]
    # T = -P[:, 3].reshape((3, 1))
    T = np.zeros((3, 1))
    T[0] = P[1, 3]
    T[1] = P[2, 3]
    T[2] = -P[0, 3]

    RT = np.hstack((R, T))
    lidar_corners = np.dot(RT, point)
    result = []
    result.append(lidar_corners[0, 0])
    result.append(lidar_corners[1, 0] + 0.7)
    result.append(lidar_corners[2, 0])
    return result


def camera_to_lidar_cnr(pts_3D, P):
    """
    convert camera corners to lidar corners
    """
    if pts_3D.shape[1] == 24:
        pts_3D = pts_3D.reshape((3, 8))

    pts_3D = np.vstack((pts_3D, np.zeros(8)))

    assert pts_3D.shape == (4, 8)

    R = np.linalg.inv(P[:, :3])
    # T = -P[:, 3].reshape((3, 1))
    T = np.zeros((3, 1))
    T[0] = -P[1, 3]
    T[1] = -P[2, 3]
    T[2] = P[0, 3]
    RT = np.hstack((R, T))

    lidar_corners = np.dot(RT, pts_3D)
    lidar_corners = lidar_corners[:3, :]

    return lidar_corners.reshape(-1, 24)


def myComputeCorners3D(Boxex3D):
    # boxex3d:h,w,l,x,y,z,ry
    #     # compute rotational matrix around yaw axis
    ry = Boxex3D[6]
    h, w, l = Boxex3D[0:3]  # 3D bounding box dimensions
    x, y, z = Boxex3D[3:6]

    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]]).reshape((3, 3))
    # 3D bounding box corners
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners = np.vstack((x_corners, y_corners, z_corners))

    # rotate and translate 3D bounding box
    corners_3D = np.dot(R, corners)
    corners_3D[0, :] = corners_3D[0, :] + x
    corners_3D[1, :] = corners_3D[1, :] + y
    corners_3D[2, :] = corners_3D[2, :] + z

    return corners_3D


def draw_gt_boxes3d(gt_boxes3d, fig, scores=None, color=(1, 1, 1), scan=None, line_width=0.05):
    DEBUG = True
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if scores is not None:
            mycolor = (0, round(scores[n], 4), 1 - round(scores[n], 4))
            # mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(0.5, 0.5, 0.5), color=(1,1,1), figure=fig)
            mlab.text3d(b[0, 0], b[0, 1], b[0, 2], '%.3f' % scores[n], scale=(0.8, 0.8, 0.8), color=mycolor, figure=fig)
        else:
            xmin = min(b[0, 0], b[6, 0])
            xmax = max(b[0, 0], b[6, 0])
            ymin = min(b[0, 1], b[6, 1])
            ymax = max(b[0, 1], b[6, 1])
            zmin = min(b[0, 2], b[6, 2])
            zmax = max(b[0, 2], b[6, 2])
            inds_inside = len(np.where(
                (scan[:, 0] >= xmin) & (scan[:, 0] <= xmax) &
                (scan[:, 1] >= ymin) & (scan[:, 1] <= ymax) &
                (scan[:, 2] >= zmin) & (scan[:, 2] <= zmax)
            )[0])
            mycolor = (0.0, 0.5, 0.5)
            mlab.text3d(b[0, 0], b[0, 1], b[0, 2], '%d' % inds_inside, scale=(0.4, 0.4, 0.4), color=mycolor, figure=fig)

            mlab.text3d(b[1, 0], b[1, 1], b[1, 2], '1', scale=(0.3, 0.3, 0.3), color=mycolor, figure=fig)
            mlab.text3d(b[2, 0], b[2, 1], b[2, 2], '2', scale=(0.3, 0.3, 0.3), color=mycolor, figure=fig)
            mlab.text3d(b[3, 0], b[3, 1], b[3, 2], '3', scale=(0.3, 0.3, 0.3), color=mycolor, figure=fig)
            mlab.text3d(b[4, 0], b[4, 1], b[4, 2], '4', scale=(0.3, 0.3, 0.3), color=mycolor, figure=fig)
            mlab.text3d(b[5, 0], b[5, 1], b[5, 2], '5', scale=(0.3, 0.3, 0.3), color=mycolor, figure=fig)
            mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '6', scale=(0.3, 0.3, 0.3), color=mycolor, figure=fig)
            mlab.text3d(b[7, 0], b[7, 1], b[7, 2], '7', scale=(0.3, 0.3, 0.3), color=mycolor, figure=fig)

        # mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=mycolor, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=mycolor, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=mycolor, tube_radius=None,
                        line_width=line_width, figure=fig)

    mlab.view(azimuth=180, elevation=None, distance=60,
              focalpoint=[12.0909996, -1.04700089, -2.03249991])  # 2.0909996 , -1.04700089, -2.03249991


def video_producer():
    img_path = '/home/hexindong/ws_dl/pyProj/cubic-local/test_result/OFHZ/'

    f = os.listdir(img_path)
    f.sort()
    videoWriter = cv2.VideoWriter('/home/hexindong/Desktop/Drive0064-Man.mp4',cv2.cv.CV_FOURCC(*'PIM1'),fps=20, frameSize=(800, 600))

    for i in f:
        img = cv2.imread(img_path + i)
        cv2.imshow('testResult',img)
        videoWriter.write(img)
        print str(i)

    videoWriter.release()

def analyse_npy_data(npy_path):
    if not os.path.isfile(npy_path):
        print ("The path: " + '{}'.format(npy_path) + "does not exit!")
        exit(0)
    else:
        data = np.load(npy_path)
    total_loss = data[0, :]
    rpn_loss_cls = data[1, :]
    rpn_loss_box = data[2, :]
    loss_cls = data[3, :]
    loss_box = data[4, :]

    plt.figure(num=1)
    labe_1 = plt.plot(np.arange(0, total_loss.size, 1, dtype=int), total_loss, label='Total_loss', color='c',
                      linewidth=1.0, linestyle='-')
    # labe_2= plt.plot(np.arange(0, total_loss.size, 1, dtype=int), rpn_loss_cls, label='Rpn_loss_cls', color='red',
    #                    linewidth=1.0, linestyle='-')
    print total_loss.size
    plt.xlim(0, total_loss.size)
    plt.ylim(0, 2)

    plt.xlabel('training times unit:10')
    plt.ylabel('error')
    # plt.xticks(np.linspace(0, 60000, 500))
    # plt.yticks(np.linspace(0, 2, 5))

    plt.title('Analyse error data')
    plt.legend()
    plt.figure(num=2)
    plt.show()


def dataset_random_index_generator():
    test_index = []
    train_index = []
    temp_index = sorted(random.sample(range(7481), 5000))
    # generate test index
    for i in range(7481):
        if i not in temp_index:
            test_index.append(i)
    # generate valid index
    valid_index = sorted(random.sample(temp_index, 1000))

    # generate train index
    for k in temp_index:
        if k not in valid_index:
            train_index.append(k)

    print train_index, test_index, valid_index

    file = open(r'train.txt', 'w')
    for j in range(len(train_index)):
        file.write(str(train_index[j]).zfill(6))
        file.write('\n')
    file.close()

    file = open(r'test.txt', 'w')
    for j in range(len(test_index)):
        file.write(str(test_index[j]).zfill(6))
        file.write('\n')
    file.close()

    file = open(r'val.txt', 'w')
    for j in range(len(valid_index)):
        file.write(str(valid_index[j]).zfill(6))
        file.write('\n')
    file.close()


def load_kitti_calib(filenames):
    """
    load projection matrix
    """
    #         P0 = np.zeros(12, dtype=np.float32)
    #         P1 = np.zeros(12, dtype=np.float32)
    #         P2 = np.zeros(12, dtype=np.float32)
    #         P3 = np.zeros(12, dtype=np.float32)
    #         R0 = np.zeros(9, dtype=np.float32)
    #         Tr_velo_to_cam = np.zeros(12, dtype=np.float32)
    #         Tr_imu_to_velo = np.zeros(12, dtype=np.float32)
    #         j = 0
    with open(filenames) as fi:
        lines = fi.readlines()
    # assert(len(lines) == 8)
    #         obj = lines[0].strip().split(' ')[1:]
    #         P0 = np.array(obj, dtype=np.float32)
    #         obj = lines[1].strip().split(' ')[1:]
    #         P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    #         obj = lines[6].strip().split(' ')[1:]
    #         P0 = np.array(obj, dtype=np.float32)
    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def draw_lidar(lidar, is_grid=True, fig=None, draw_axis=True):
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))

    mlab.points3d(pxs, pys, pzs, prs,
                  mode='point',  # 'point'  'sphere'
                  colormap='gnuplot',  # 'bone',  #'spectral',  #'copper',
                  scale_factor=1,
                  figure=fig)
    # draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        for y in np.arange(-40, 40, 1):
            x1, y1, z1 = -40.0, float(y), -1.5
            x2, y2, z2 = 40.0, float(y), -1.5
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.1, 0.1, 0.1), tube_radius=None, line_width=0.1,
                        figure=fig)

        for x in np.arange(-40, 40, 1):
            x1, y1, z1 = float(x), -40.0, -1.5
            x2, y2, z2 = float(x), 40.0, -1.5
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.1, 0.1, 0.1), tube_radius=None, line_width=0.1,
                        figure=fig)

    # draw axis
    if draw_axis:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        axes = np.array([
            [2., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 2., 0.],
        ], dtype=np.float64)
        fov = np.array([
            [40., 40., 0., 0.],
            [40., -40., 0., 0.],
        ], dtype=np.float64)

        mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                    figure=fig)
        mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                    figure=fig)

    mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=None, distance=50,
              focalpoint=[12.0909996, -1.04700089, -2.03249991])  # 2.0909996 , -1.04700089, -2.03249991


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def data_show(path, index=21):
    # calibration ============================
    filename = os.path.join(path, 'calib', str(index).zfill(6) + '.txt')
    calib = load_kitti_calib(filename)
    Tr = calib['Tr_velo2cam']

    # ====gt_box=====
    filename = os.path.join(path, 'label_2', str(index).zfill(6) + '.txt')
    with open(filename, 'r') as file_txt:
        gt = file_txt.readlines()
    box_cam = np.zeros((len(gt), 7), dtype=np.float32)
    Liadr_cnr = np.zeros((len(gt), 24), dtype=np.float32)
    Camera_cnr = np.zeros((len(gt), 24), dtype=np.float32)
    # Imgae
    filename = os.path.join(path, 'image_2', str(index).zfill(6) + '.png')
    image_rgb = cv2.imread(filename)
    # plt.subplot(311)
    plt.imshow(image_rgb)
    plt.show()
    index_car = 0
    for idx in range(len(gt)):
        if str.lower(gt[idx].split()[0]) == 'car':
            box_cam[index_car, :] = np.array(map(eval, gt[idx].split()[8:15]))
            single_cam_cnr = myComputeCorners3D(box_cam[index_car, :])
            Camera_cnr[idx, :] = single_cam_cnr.reshape(24)
            Liadr_cnr[index_car, :] = camera_to_lidar_cnr(single_cam_cnr, Tr)
            index_car += 1
    Liadr_all_cnr = Liadr_cnr[:, :24].reshape((-1, 3, 8)).transpose((0, 2, 1))
    Camera_all_cnr = Camera_cnr[:, :24].reshape((-1, 3, 8)).transpose((0, 2, 1))
    # LiDar data ============================
    filename = os.path.join(path, 'velodyne', str(index).zfill(6) + '.bin')
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(scan, fig=fig)
    draw_gt_boxes3d(Liadr_all_cnr, fig, color=(1, 1, 1), line_width=1, scan=scan)
    image_pcl = mlab.screenshot()
    mlab.show()
    # mlab.close()



    # # Lidae bird view
    # filename = os.path.join(path, 'lidar_bv', str(index).zfill(6) + '.npy')
    # bv = np.load(filename)
    # image_bv = scale_to_255(bv[:, :, 8], min=0, max=2)
    # # plt.subplot(312)
    # plt.imshow(image_bv)
    # plt.show()

    # plt.subplot(313)
    # plt.imshow(image_pcl)
    # plt.show()


def modify_result(txtpath, pklpath):
    if not os.path.exists(txtpath):
        print "The path: {} doesn't exits! ".format(txtpath)
        return 0
    inf = pickle.load(open(pklpath))
    files = sorted(os.listdir(txtpath))

    for f_idx, f in enumerate(files):
        writingLines = []
        name = os.path.join(txtpath, f)
        with open(name, 'r') as f1:
            lines = f1.readlines()
        box = []
        for i in range(len(inf[1][f_idx])):
            box.append(cal_cnr2box(inf[1][f_idx][i][0:24]))

        for idx, l in enumerate(lines):
            oneline = l.split()
            oneline[8] = str(box[idx][0])  # h
            oneline[9] = str(box[idx][1])  # w
            oneline[10] = str(box[idx][2])  # l
            oneline[11] = str(box[idx][3])  # x
            oneline[12] = str(box[idx][4])  # y
            oneline[13] = str(box[idx][5])  # z
            oneline[14] = str(box[idx][6])  # ry
            l1 = ''
            for word in oneline:
                l1 += word + ' '
            l1 += '\n'
            writingLines.append(l1)
        with open(name, 'w') as f2:
            f2.writelines(writingLines)
        print  f + 'has been rewrited\n'


def cal_cnr2box(cnr):
    corner = np.array(cnr, dtype=np.float32).reshape((3, 8))
    box3d = np.zeros(7)
    box3d[0] = round(np.sqrt(np.sum(np.square(corner[:, 0] - corner[:, 4]))), 2)  # h 0:4
    box3d[1] = round(np.sqrt(np.sum(np.square(corner[:, 0] - corner[:, 1]))), 2)  # w 0:1
    box3d[2] = round(np.sqrt(np.sum(np.square(corner[:, 0] - corner[:, 3]))), 2)  # l 0:3

    box3d[3:6] = corner.mean(1)
    box3d[3:6] = lidar_point_to_camera(box3d[3:6])
    box3d[3] = round(box3d[3], 2)  # x
    box3d[4] = round(box3d[4], 2)  # y
    box3d[5] = round(box3d[5], 2)  # z

    box3d[6] = -np.arctan2(corner[0, 1] - corner[0, 0], corner[1, 0] - corner[1, 1]) - 1.5707963
    if box3d[6] > 3.1415926:
        box3d[6] -= 3.1415926
    if box3d[6] < -3.1415926:
        box3d[6] += 3.1415926
    box3d[6] = round(box3d[6], 2)

    return box3d


def display_stiData(Scan):
    point_cloud = Scan.reshape((16, 2016, 7))
    pointx = point_cloud[:, :, 0].flatten()
    pointy = point_cloud[:, :, 1].flatten()
    pointz = point_cloud[:, :, 2].flatten()
    intensity = point_cloud[:, :, 3].flatten()
    labels = point_cloud[:, :, 6].flatten()

    seg_point = PointCloud()
    seg_point.header.frame_id = 'rslidar'
    channels1 = ChannelFloat32()
    seg_point.channels.append(channels1)
    seg_point.channels[0].name = "rgb"
    channels2 = ChannelFloat32()
    seg_point.channels.append(channels2)
    seg_point.channels[1].name = "intensity"

    for i in range(32256):
        seg_point.channels[1].values.append(intensity[i])
        if labels[i] == 1:
            seg_point.channels[0].values.append(255)
            geo_point = Point32(pointx[i], pointy[i], pointz[i])
            seg_point.points.append(geo_point)
        else:
            seg_point.channels[0].values.append(255255255)
            geo_point = Point32(pointx[i], pointy[i], pointz[i])
            seg_point.points.append(geo_point)
            # elif result[i] == 2:
            #     seg_point.channels[0].values.append(255255255)
            #     geo_point = Point32(pointx[i], pointy[i], pointz[i])
            #     seg_point.points.append(geo_point)
            # elif result[i] == 3:
            #     seg_point.channels[0].values.append(255000)
            #     geo_point = Point32(pointx[i], pointy[i], pointz[i])
            #     seg_point.points.append(geo_point)

    return seg_point


def files2npy():
    path = '/home/hexindong/DATASET/kittidataset/KITTI/object/test'
    nameList = sorted(os.listdir(path + '/velodyne'))
    for n in nameList:
        label = np.fromfile(path + '/velodyne/' + n, dtype=np.float32).reshape((-1, 4))
        np.save(path + '/velodyne-npy/' + n.split('.')[0], label)  # ,fmt='%4.2f'
    pass


def anchor_target_layer(gt_boxes, gt_boxes_3d, im_info, _feat_stride=[16, ], anchor_scales=[8, 16, 32]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    DEBUG = False
    _anchors = np.array(
        [[-19, -8, 20, 8],
         [-5, -2, 5, 3],
         [-8, -19, 8, 20],
         [-2, -5, 3, 5]])
    _num_anchors = _anchors.shape[0]

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0
    # map of shape (..., H, W)
    # height, width = rpn_cls_score.shape[1:3]


    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap



    # map of shape (..., H, W)
    height, width = 8, 6

    _feat_stride = 8
    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    a = _anchors.reshape((1, A, 4))
    b = shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = (a + b)
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < 500 + _allowed_border) &  # width
        (all_anchors[:, 3] < 500 + _allowed_border)  # height
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them

        # hard negative for proposal_target_layer
        hard_negative = np.logical_and(0 < max_overlaps, max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP)
        labels[hard_negative] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # random sample

    # fg label: above threshold IOU
    # print np.where(max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP)
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1


        # print "was %s inds, disabling %s, now %s inds" % (
        # len(bg_inds), len(disable_inds), np.sum(labels == 0))

    anchors_3d = bv_anchor_to_lidar(anchors)
    bbox_targets = _compute_targets_3d(anchors_3d, gt_boxes_3d[argmax_overlaps, :])

    # print 'labels = 0:, ', np.where(labels == 0)
    all_inds = np.where(labels != -1)
    labels_new = labels[all_inds]
    zeros = np.zeros((labels_new.shape[0], 1), dtype=np.float32)
    anchors = np.hstack((zeros, anchors[all_inds])).astype(np.float32)
    anchors_3d = np.hstack((zeros, anchors_3d[all_inds])).astype(np.float32)

    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # labels[hard_negative] = -1
    # # subsample negative labels if we have too many
    # num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    # bg_inds = np.where(labels != 1)[0]
    # # print len(bg_inds)
    # if len(bg_inds) > num_bg:
    #     disable_inds = npr.choice(
    #         bg_inds, size=(num_bg), replace=False)
    #     labels[disable_inds] = 0

    # all_inds = np.where(labels != -1)
    # labels_new = labels[all_inds]
    # zeros = np.zeros((labels_new.shape[0], 1), dtype=np.float32)
    # # print zeros.shape
    # # print len(all_inds)
    # anchors =  np.hstack((zeros, anchors[all_inds])).astype(np.float32)
    # anchors_3d =  np.hstack((zeros, anchors_3d[all_inds])).astype(np.float32)

    # bg_inds = np.where(hard_negative == True)[0]
    # disable_inds = npr.choice(
    #         bg_inds, size=(len(bg_inds)/2.), replace=False)
    # labels[disable_inds] = -1


    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means:'
        print means
        print 'stdevs:'
        print stds

    if DEBUG:
        print 'gt_boxes_3d: ', gt_boxes_3d[argmax_overlaps, :].shape
        print 'labels shape before unmap: ', labels.shape
        print 'targets shaoe before unmap: ', bbox_targets.shape
    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlap', np.max(max_overlaps)
        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count
        print 'fg inds: ', fg_inds
        print 'label shape', labels.shape
        print 'bbox_targets', bbox_targets.shape

    # labels
    rpn_labels = labels
    rpn_bbox_targets = bbox_targets

    if DEBUG:
        print 'labels shape: ', labels.shape
        print 'targets shape: ', bbox_targets.shape

    return rpn_labels, rpn_bbox_targets, anchors, anchors_3d


def rpn_classify_tf(lidar3d, rpn, im_info):
    pred_box = rpn[1][1:7]
    pred_scores = rpn[1][0, :].reshape(-1, 1)
    ToClassifyPnts = [np.empty([]) for _ in range(pred_box.shape[0])]
    for i, box in enumerate(pred_box):
        indice = check_inside(lidar3d, box)
        ToClassifyPnts[i] = lidar3d[indice, :]

    np.hstack([pred_scores, ToClassifyPnts])
    return ToClassifyPnts


def check_inside(lidar, pred_box):
    # pred_box [x,y,z,l,w,h]
    xmin = pred_box[0] - pred_box[3] / 2.
    xmax = pred_box[0] + pred_box[3] / 2.
    ymin = pred_box[1] - pred_box[4] / 2.
    ymax = pred_box[1] + pred_box[4] / 2.
    zmin = pred_box[2] - pred_box[5] / 2.
    zmax = pred_box[2] + pred_box[5] / 2.

    inds_inside = len(np.where(
        (lidar[:, 0] >= xmin) & (lidar[:, 0] <= xmax) &
        (lidar[:, 1] >= ymin) & (lidar[:, 1] <= ymax) &
        (lidar[:, 2] >= zmin) & (lidar[:, 2] <= zmax)
    )[0])
    return inds_inside

def rename_bat(path):
    if not os.path.exists(path):
        print "The path: {} doesn't exits! ".format(path)
        return 0
    files = sorted(os.listdir(path),key=lambda name:int(name[18:-4]))
    for n in range(len(files)):
        oldname = os.path.join(path, files[n])
        newname = os.path.join(path, "170829-1744-LM120_{}".format(n+2700) + '.pcd')
        os.rename(oldname, newname)
        print(oldname, '======>', newname)


if __name__ == '__main__':
    # path ='/home/hexindong/DATASET/stidataset/LM120-170829-1743/pcd/'
    # rename_bat(path)
    # libel_fname= '/home/hexindong/DATASET/stidataset/LM120-170829-1743/label/result.txt'
    # new_lines=[]
    # with open(libel_fname, 'r') as f:
    #     lines = f.readlines()
    # for line in lines:
    #     line = line.replace('LM120-170829-1743','170829-1744-LM120')
    #     new_lines.append(line)
    # f2 = open('/home/hexindong/DATASET/stidataset/LM120-170829-1743/label/result_mo.txt', 'w')
    # for line in new_lines:
    #     f2.write(line)
    # f2.close()
    # print 'Done !'

    # rospy.init_node('rostensorflow')
    # pub = rospy.Publisher('prediction', PointCloud, queue_size=1000)
    # rospy.loginfo("ROS begins ...")
    # #
    # idx = 0
    # while True:
    #     base_path = "/home/hexindong/DATASET/stidataset"
    #     filename = os.path.join(base_path,str(idx)+ '.txt')
    #     scan = np.loadtxt(filename)
    #     pointcloud = display_stiData(scan)
    #     pub.publish(pointcloud)
    #     # points = scan[:,0:4]
    #     # fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
    #     # draw_lidar(points, fig=fig)
    #     # mlab.show()
    #     idx += 1
    #     if idx > 42: idx = 0
    video_producer()
    print 'done~'
    # if 1:
    #     while True:
    #         # import tensorflow as tf
    #         # a = tf.constant(0,tf.float32,[1,2])
    #         idx = input('Type a new index: ')
    #         data_show('/home/hexindong/ws_dl/pyProj/CubicNet-server/data/training/', idx)
    #
    #         # for i in range(20):
    #         #     filepath = "/home/hexindong/ws_dl/pyProj/MV3D_TF/data/KITTI_ORIGIN/object/training/"
    #         #     filename=filepath+"velodyne/"+str(i).zfill(6) + ".bin"
    #         #     print("Processing: ", filename)
    #         #     scan = np.fromfile(filename, dtype=np.float32)
    #         #     scan = scan.reshape((-1, 4))
    #         #     front_view = point_cloud_to_panorama(scan)
    #         #
    #         #     #save
    #         #     savepath=filepath+"lidar_fv/"+str(i).zfill(6) + ".npy"
    #         #     np.save(savepath,front_view)
    #         #
    #         #     # filename = os.path.join(path, 'lidar_bv', str(index).zfill(6) + '.npy')
    #         #     # bv = np.load(filename)
    #         #     # image_fv = scale_to_255(front_view[:, :, 8], min=0, max=2)
    #         #     # plt.subplot(312)
    #         #     plt.imshow(front_view)
    #         #     plt.show()
