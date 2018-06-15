
import numpy as np
from numpy import random
import tensorflow as tf
from tools.data_visualize import pcd_vispy,pcd_show_now,boxary2dic
from network.config import cfg
from tensorflow.python.ops import init_ops

DEBUG = False

cubic_size = [cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 1]#TODO: cubic feature define1,2,4,or else?

def cubic_rpn_grid_pyfc(lidarPoints, rpnBoxes,method):
    # rpnBoxes:(x1,y1,z1),(x2,y2,z2),cls_label,yaw
    res = []
    display_stack=[]
    if DEBUG:
        pass
        display_stack.append(pcd_vispy(lidarPoints, boxes=boxary2dic(rpnBoxes),visible=False,multi_vis=True))
    rpn_new_yaw=[]
    for iidx,box in enumerate(rpnBoxes):
        rpn_points,min_vertex,ctr_vertex = bounding_filter(lidarPoints,box)
        points_mv_min = np.subtract(rpn_points,min_vertex)  # using fot coordinate
        points_mv_ctr = np.subtract(rpn_points,ctr_vertex)  # using as feature
        if method == 'train' and cfg.TRAIN.USE_AUGMENT_IN_CUBIC_GEN:
            if DEBUG:
                angel = 0
            else:
                angel = (random.rand()-0.500) * np.pi * 0.9# *np.pi*2 #[0~360] #counter clockwise rotation#TODO: check ;to change 0.01->big scalar
            scalar = 1.05 - random.rand()*0.1
            translation = np.random.rand(3, 1) * 0.01
            points_mv_ctr_rot_nobound = rot_sca_pc(points_mv_ctr, angel, scalar, translation)
            points_mv_ctr_rot, min_p, ctr_p = bounding_filter(points_mv_ctr_rot_nobound, [0, 0, 0])

            x_cub = np.divide(points_mv_ctr_rot[:, 0] - min_p[0], cfg.CUBIC_RES[0]).astype(np.int32)
            y_cub = np.divide(points_mv_ctr_rot[:, 1] - min_p[1], cfg.CUBIC_RES[1]).astype(np.int32)
            z_cub = np.divide(points_mv_ctr_rot[:, 2] - min_p[2], cfg.CUBIC_RES[2]).astype(np.int32)
            if not DEBUG:
                # feature = np.hstack((np.ones([len(points_mv_ctr_rot[:,3]),1]),points_mv_ctr_rot[:,3].reshape(-1,1))) #points_mv_ctr_rot
                feature = np.ones([len(points_mv_ctr_rot[:,3]),1], dtype=np.float32)  # points_mv_ctr_rot
            else:
                # feature = np.hstack((x_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[0]/2),y_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[1]/2),z_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[2]/2),points_mv_ctr_rot[:,3].reshape(-1,1))) #points_mv_ctr_rot
                feature = np.hstack((x_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[0]/2),y_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[1]/2),z_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[2]/2))) #points_mv_ctr_rot
        else:  # method: test
            angel = 0.0
            x_cub = np.divide(points_mv_min[:, 0], cfg.CUBIC_RES[0]).astype(np.int32)
            y_cub = np.divide(points_mv_min[:, 1], cfg.CUBIC_RES[1]).astype(np.int32)
            z_cub = np.divide(points_mv_min[:, 2], cfg.CUBIC_RES[2]).astype(np.int32)
            # feature = np.hstack((np.ones([len(points_mv_ctr[:,3]),1]),points_mv_ctr[:,3:]))
            feature = np.ones([len(points_mv_ctr[:,3]),1],dtype=np.float32)

        rpn_new_yaw.append(angel)  # gt_yaw - rotation: because gt is clockwise and rotation is counter clockwise#TODO hxd:check
        cubic_feature = np.zeros(shape=cubic_size, dtype=np.float32)
        cubic_feature[x_cub, y_cub, z_cub] = feature  # TODO:select&add feature # points_mv_ctr  # using center coordinate system
        res.append(cubic_feature)

        if DEBUG:
            box_mv = [box[0] - box[0], box[1] - box[1], box[2] - box[2],cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1],cfg.CUBIC_SIZE[2],1,0,0]
            box_gt_mv = [box[0] - box[0], box[1] - box[1], box[2] - box[2], cfg.ANCHOR[0], cfg.ANCHOR[1], cfg.ANCHOR[2],1, 0, 0]

            display_stack.append(pcd_vispy(cubic_feature.reshape(-1, 3),name='grid_'+str(iidx), boxes=boxary2dic(np.array(box_mv)),visible=False,point_size =0.1,multi_vis=True))
            display_stack.append(pcd_vispy(points_mv_ctr.reshape(-1, 4),name='origin_'+str(iidx), boxes=boxary2dic(np.array(box_gt_mv)),visible=False,point_size =0.1,multi_vis=True))
        # break
    if DEBUG:
        pcd_show_now()
    stack_size = np.concatenate((np.array([-1]), cubic_size))
    return np.array(res, dtype=np.float32).reshape(stack_size),np.array(rpn_new_yaw,dtype=np.float32)

def rot_sca_pc(points, rotation,scalar,translation):
    # points: numpy array;  translation: moving scalar which should be small
    R = np.array([[np.cos(rotation), -np.sin(rotation), 0.],
                  [np.sin(rotation), np.cos(rotation), 0.],
                  [0, 0, 1]], dtype=np.float32)
    assert translation.shape ==(3,1), 'File rpn_3dcnn Function rot_sca_pc :T is  incompatible with transform'
    points_rot = np.matmul(R, points[:, 0:3].transpose()) + translation
    points_rot_sca = points_rot*scalar
    return np.hstack((points_rot_sca.transpose(),points[:,3:]))

def bounding_filter(points,box):

    x_min = box[0] - float(cfg.ANCHOR[0]) / 2
    x_max = box[0] + float(cfg.ANCHOR[0]) / 2
    y_min = box[1] - float(cfg.ANCHOR[1]) / 2
    y_max = box[1] + float(cfg.ANCHOR[1]) / 2
    z_min = box[2] - float(cfg.ANCHOR[2]) / 2
    z_max = box[2] + float(cfg.ANCHOR[2]) / 2

    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    f_filt = np.logical_and((x_points > x_min), (x_points < x_max))
    s_filt = np.logical_and((y_points > y_min), (y_points < y_max))
    z_filt = np.logical_and((z_points > z_min), (z_points < z_max))
    fliter = np.logical_and(np.logical_and(f_filt, s_filt), z_filt)
    indice = np.flatnonzero(fliter)
    filter_points = points[indice]

    return filter_points,np.array([x_min,y_min,z_min,0.],dtype=np.float32),np.array([box[0],box[1],box[2],0.],dtype=np.float32)


class cubic(object):
    def __init__(self, channel,training=True):
        with tf.variable_scope('conv3d_1', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_1 = tf.layers.Conv3D(filters=channel[1], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[1, 1, 1], padding="valid", _reuse=tf.AUTO_REUSE,
                                             kernel_initializer=init_ops.variance_scaling_initializer,
                                             _scope=scope, trainable=training)
            self.maxpool_1 = tf.layers.MaxPooling3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')
            self.bn_1 = tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        with tf.variable_scope('conv3d_2', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_2 = tf.layers.Conv3D(filters=channel[2], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[1, 1, 1], padding="valid", _reuse=tf.AUTO_REUSE,
                                             kernel_initializer=init_ops.variance_scaling_initializer,
                                             _scope=scope, trainable=training)
            self.maxpool_2 = tf.layers.MaxPooling3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')
            self.bn_2 = tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        with tf.variable_scope('conv3d_3', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_3 = tf.layers.Conv3D(filters=channel[3], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[1, 1, 1], padding="valid", _reuse=tf.AUTO_REUSE,
                                             kernel_initializer=init_ops.variance_scaling_initializer,
                                             _scope=scope, trainable=training)
            self.bn_3 = tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        with tf.variable_scope('fc_bn_1', reuse=tf.AUTO_REUSE) as scope:
            self.dense_1 = tf.layers.Dense(channel[4], tf.nn.relu, _reuse=tf.AUTO_REUSE, _scope=scope,
                                           kernel_initializer=init_ops.variance_scaling_initializer)
            self.bn_4 = tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        with tf.variable_scope('fc_2', reuse=tf.AUTO_REUSE) as scope:
            self.dense_2 = tf.layers.Dense(channel[5], _reuse=tf.AUTO_REUSE, _scope=scope,
                                           kernel_initializer=init_ops.variance_scaling_initializer
                                           )

    def apply(self, inputs):
        out_conv3d_1 = self.conv3d_1.apply(inputs)
        out_maxp_1 = self.maxpool_1.apply(out_conv3d_1)
        # out_bn_1=self.bn_1.apply(out_maxp_1)

        out_conv3d_2 = self.conv3d_2.apply(out_maxp_1)
        out_maxp_2 = self.maxpool_2.apply(out_conv3d_2)
        # out_bn_2=self.bn_2.apply(out_conv3d_2)

        out_conv3d_3 = self.conv3d_3.apply(out_maxp_2)
        # out_bn_3=self.bn_3.apply(out_conv3d_3)

        conv3d_flatten = tf.layers.flatten(out_conv3d_3)

        dense_out_1 = self.dense_1.apply(conv3d_flatten)
        # dense_bn_1 = self.bn_4.apply(dense_out_1)

        res = self.dense_2.apply(dense_out_1)

        return res  # tf.convert_to_tensor(res_stack, dtype=tf.float32)


if __name__ == '__main__':
    from dataset.dataset import dataset_STI_train, dataset_KITTI_train
    from easydict import EasyDict as edict

    arg = edict()
    arg.imdb_type = 'kitti'

    dataset = dataset_KITTI_train(arg)
    DEBUG=True
    cubic_size = [cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 3]
    while True:
        idx = input('Type a new index: ')
        blobs = dataset.get_minibatch(idx)
        cubic_rpn_grid_pyfc(blobs['lidar3d_data'], blobs['gt_boxes_3d'],method='train')
