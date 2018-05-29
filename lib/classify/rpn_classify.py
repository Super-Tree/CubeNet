import os
import numpy as np
from network.config import cfg
import numpy.random as npr
from tools.data_visualize import draw_3dPoints_box

DEBUG = True

def rpn_serial_extract_tf(lidar3d, rpn, im_info=None):

    pred_box = rpn[:, 1:7]  # blob_3d_box x,y,z,l,w,h
    ToClassifyPoints = np.zeros([1, 8], dtype=np.float32)  # to start vstack and iteration
    for i, boxes in enumerate(pred_box):
        indices = check_inside(lidar3d, boxes)
        if indices.size:  # in case of empty box
            boundingPoints = lidar3d[indices, :]
            featurePoints = boundingPoints - [boxes[0], boxes[1], boxes[2], 0]
            points = np.hstack([np.ones((indices.shape[0], 1))*i, boundingPoints, featurePoints[:, 0:3]])
            ToClassifyPoints = np.vstack([ToClassifyPoints, points])

    return ToClassifyPoints[1:, :].astype(np.float32)

def rpn_regular_extract_tf(lidar3d, rpn, im_info=None):

    pred_box = rpn[:, 1:7]  # blob_3d_box
    box_cnt = pred_box.shape[0]
    ToClassifyPoints = np.zeros([box_cnt, cfg.RPN_POINTS_REMAIN, 4], dtype=np.float32)
    # TODO: for deal with variable density of points(1000~60),we take two schemes
    # for i, boxes in enumerate(pred_box):
    #     indices = check_inside(lidar3d, boxes)
    #     if indices.shape[0] > cfg.RPN_POINTS_REMAIN:
    #         indices = npr.choice(indices, size=cfg.RPN_POINTS_REMAIN, replace=False)
    #     ToClassifyPoints[i, 0:indices.shape[0]] = lidar3d[indices, :]

    for i, boxes in enumerate(pred_box):
        indices = check_inside(lidar3d, boxes)
        indices = npr.choice(indices, size=cfg.RPN_POINTS_REMAIN)
        ToClassifyPoints[i, 0:indices.shape[0]] = lidar3d[indices, :]

    return ToClassifyPoints


def check_inside(lidar, pred_box):
    # pred_box [x,y,z,l,w,h]
    xmin = pred_box[0] - pred_box[3]/2.
    xmax = pred_box[0] + pred_box[3]/2.
    ymin = pred_box[1] - pred_box[4]/2.
    ymax = pred_box[1] + pred_box[4]/2.
    zmin = pred_box[2] - pred_box[5]/2.
    zmax = pred_box[2] + pred_box[5]/2.

    inds_inside = np.where(
        (lidar[:, 0] >= xmin) & (lidar[:, 0] <= xmax) &
        (lidar[:, 1] >= ymin) & (lidar[:, 1] <= ymax) &
        (lidar[:, 2] >= zmin) & (lidar[:, 2] <= zmax)
    )[0]
    return inds_inside


if __name__ == '__main__':
    index = 32  # input('Type a new index: ')
    path = '/home/hexindong/DATASET/kittidataset/KITTI/object/train/'
    filename = os.path.join(path, 'velodyne', str(index).zfill(6) + '.bin')
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    box = np.array([[0.8,5,5,-1.,3,3,3],[0.9,7,7,-1.,3,3,3]],dtype=np.float32)
    res = rpn_serial_extract_tf(scan, box)

    if DEBUG:
        draw_3dPoints_box(res[:,1:5], Boxex3D=box)
