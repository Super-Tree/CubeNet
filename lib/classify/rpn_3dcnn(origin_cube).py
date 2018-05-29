
import numpy as np
from numpy import random
import tensorflow as tf
from tools.data_visualize import pcd_vispy,pcd_show_now
from network.config import cfg
from easydict import EasyDict as edict
from dataset.dataset import dataset_STI_train

DEBUG = False

shape = lambda i: int(np.ceil(np.round(cfg.ANCHOR[i] / cfg.CUBIC_RES[i], 3)))  # Be careful about python number  decimal
#TODO: cubic feature define 2,4,or else?
cubic_size = [shape(0), shape(1), shape(2),2]


def cubic_rpn_grid_pyfc(lidarPoints, rpnBoxes,method):
    res = []
    display_stack=[]
    if DEBUG:
        print 'Start vispy ...'
        display_stack.append(pcd_vispy(lidarPoints, boxes=rpnBoxes,visible=False,multi_vis=True))

    for box in rpnBoxes:
        rpn_points,min_vertex,ctr_vertex = bounding_filter(lidarPoints,box)
        points_mv_min = np.subtract(rpn_points,min_vertex)  # using fot coordinate
        points_mv_ctr = np.subtract(rpn_points,ctr_vertex)  # using as feature

        x_cub = np.divide(points_mv_min[:, 0], cfg.CUBIC_RES[0]).astype(np.int32)
        y_cub = np.divide(points_mv_min[:, 1], cfg.CUBIC_RES[1]).astype(np.int32)
        z_cub = np.divide(points_mv_min[:, 2], cfg.CUBIC_RES[2]).astype(np.int32)
        feature = np.hstack((np.ones([len(points_mv_ctr[:,3]),1]),points_mv_ctr[:,3:]))

        cubic_feature = np.zeros(shape=cubic_size, dtype=np.float32)
        cubic_feature[x_cub, y_cub, z_cub] =feature# TODO:select&add feature # points_mv_ctr  # using center coordinate system
        res.append(cubic_feature)

        if DEBUG:
            box_mv = [box[0] - box[0], box[1] - box[1], box[2] - box[2],shape(0), shape(1),shape(2),box[6],box[7],0]
            # box_mv = [box[0] - box[0], box[1] - box[1], box[2] - box[2],cfg.ANCHOR[0], cfg.ANCHOR[1],
            #           cfg.ANCHOR[2], box[6],box[7],0]
            display_stack.append(pcd_vispy(cubic_feature.reshape(-1, 4), boxes=np.array(box_mv),visible=False,multi_vis=True))
            display_stack.append(pcd_vispy(points_mv_ctr.reshape(-1, 4), boxes=np.array(box_mv),visible=False,multi_vis=True))
        break
    if DEBUG:
        pcd_show_now()
    stack_size = np.concatenate((np.array([-1]), cubic_size))
    return np.array(res, dtype=np.float32).reshape(stack_size)

def rot_sca_pc(points, rotation,scalar,translation):
    # points: numpy array;translation: moving scalar which should be small
    R = np.array([[np.cos(rotation), -np.sin(rotation), 0.],
                  [np.sin(rotation), np.cos(rotation), 0.],
                  [0, 0, 1]], dtype=np.float32)
    assert translation.shape ==(3,1), 'File rpn_3dcnn Function rot_sca_pc :T is  incompatible with transform'
    # T = np.random.randn(3, 1) * translation
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
    def __init__(self, batch_size, channel,training=True):
        self.batch_size = batch_size
        with tf.variable_scope('conv3d_1', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_1 = tf.layers.Conv3D(filters=channel[0], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[2, 2, 2], padding="valid", _reuse=tf.AUTO_REUSE,
                                             _scope=scope, trainable=training)
            self.bn_1 = tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        with tf.variable_scope('conv3d_2', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_2 = tf.layers.Conv3D(filters=channel[1], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[1, 1, 1], padding="valid", _reuse=tf.AUTO_REUSE,
                                             _scope=scope, trainable=training)
            # self.maxpool_1= tf.layers.MaxPooling3D(pool_size=[2,2,2],strides=[2,2,2],padding='same')
            self.bn_2 = tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        with tf.variable_scope('conv3d_3', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_3 = tf.layers.Conv3D(filters=channel[2], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[1, 1, 1], padding="valid", _reuse=tf.AUTO_REUSE,
                                             _scope=scope, trainable=training)
            self.bn_3=tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        with tf.variable_scope('fc_bn_1', reuse=tf.AUTO_REUSE) as scope:
                self.dense_1 = tf.layers.Dense(channel[3], tf.nn.relu, _reuse=tf.AUTO_REUSE, _scope=scope)
                self.bn_4 = tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        with tf.variable_scope('fc_2', reuse=tf.AUTO_REUSE) as scope:
                self.dense_2 = tf.layers.Dense(channel[4], _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs):
        conv3d_input = tf.reshape(inputs, np.concatenate((np.array([-1]), cubic_size)))
        out_conv3d_1 = self.conv3d_1.apply(conv3d_input)
        out_bn_1=self.bn_1.apply(out_conv3d_1)
        out_conv3d_2 = self.conv3d_2.apply(out_bn_1)
        out_bn_2=self.bn_2.apply(out_conv3d_2)
        out_conv3d_3 = self.conv3d_3.apply(out_bn_2)
        out_bn_3=self.bn_3.apply(out_conv3d_3)

        conv3d_flatten = tf.layers.flatten(out_bn_3)

        dense_out_1 = self.dense_1.apply(conv3d_flatten)
        dense_bn_1 = self.bn_4.apply(dense_out_1)
        res = self.dense_2.apply(dense_bn_1)

        return res  #,tf.convert_to_tensor(res_stack, dtype=tf.float32)

    def apply2(self, inputs):
        inputs = tf.reshape(inputs, cubic_size)
        out_conv3d_1 = self.conv3d_1.apply(inputs)
        out_conv3d_2 = self.conv3d_2.apply(out_conv3d_1)
        out_conv3d_3 = self.conv3d_3.apply(out_conv3d_2)

        mid_shape = tf.shape(out_conv3d_3)
        res = out_conv3d_3
        return res


if __name__ == '__main__':
    arg = edict()
    arg.imdb_type = 'sti'

    dataset = dataset_STI_train(arg)
    DEBUG=True
    while True:

        idx = input('Type a new index: ')
        blobs = dataset.get_minibatch(idx)
        cubic_rpn_grid_pyfc(blobs['lidar3d_data'], blobs['gt_boxes_3d'],method='train')
