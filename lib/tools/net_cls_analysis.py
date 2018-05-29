# coding=utf-8
from numpy import random
import os
import math
import numpy as np
import tensorflow as tf
from tools.timer import Timer
from network.config import cfg
from tensorflow.python import pywrap_tensorflow
from tools.data_visualize import pcd_vispy, vispy_init
from dataset.dataset import dataset_STI_train, dataset_KITTI_test,dataset_KITTI_train
from easydict import EasyDict as edict
from tools.data_visualize import pcd_vispy, pcd_show_now

import matplotlib.pyplot as plt
#
# from multiprocessing import Process,Queue
# MSG_QUEUE = Queue(200)

DEBUG = False
BATCH_CNT = 2
shape = lambda i: int(np.ceil(np.round(cfg.ANCHOR[i] / cfg.CUBIC_RES[i], 3)))  # Be careful about python number  decimal
cubic_size = [shape(0), shape(1), shape(2), 2]
cubic_show_size = [shape(0), shape(1), shape(2), 4]
cubic_batch_size = [BATCH_CNT, shape(0), shape(1), shape(2), 2]

class cubic(object):
    def __init__(self, batch_size, channel, training=True):
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
            self.bn_3 = tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        with tf.variable_scope('fc_bn_1', reuse=tf.AUTO_REUSE) as scope:
            self.dense_1 = tf.layers.Dense(channel[3], tf.nn.relu, _reuse=tf.AUTO_REUSE, _scope=scope)
            self.bn_4 = tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        with tf.variable_scope('fc_2', reuse=tf.AUTO_REUSE) as scope:
            self.dense_2 = tf.layers.Dense(channel[4], _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs):
        conv3d_input = tf.reshape(inputs, np.concatenate((np.array([-1]), cubic_size)))
        out_conv3d_1 = self.conv3d_1.apply(conv3d_input)
        out_bn_1 = self.bn_1.apply(out_conv3d_1)
        out_conv3d_2 = self.conv3d_2.apply(out_bn_1)
        out_bn_2 = self.bn_2.apply(out_conv3d_2)
        out_conv3d_3 = self.conv3d_3.apply(out_bn_2)
        out_bn_3 = self.bn_3.apply(out_conv3d_3)

        conv3d_flatten = tf.layers.flatten(out_bn_3)

        dense_out_1 = self.dense_1.apply(conv3d_flatten)
        dense_bn_1 = self.bn_4.apply(dense_out_1)
        res = self.dense_2.apply(dense_bn_1)

        return res

class CubicNet_Train(object):
    def __init__(self):
        self.weights = '/home/hexindong/ws_dl/pyProj/CubicNet-server/output/msfg/CubicNet_iter_108402.ckpt'
        arg = edict()
        arg.imdb_type = 'kitti'
        arg.use_demo = True
        self.dataset = dataset_KITTI_test(arg)
        self.cube_batch = tf.placeholder(tf.float32, shape=cubic_batch_size, name='cubes')
        with tf.variable_scope('cubic_cnn', reuse=tf.AUTO_REUSE) as scope:
            self.cubic_3dcnn = cubic(BATCH_CNT, [64, 128, 128, 64, 2])

        self.result = self.cubic_3dcnn.apply(self.cube_batch)
        self.saver = tf.train.Saver(max_to_keep=100)

    def training(self, sess):
        sess.run(tf.global_variables_initializer())
        reader = pywrap_tensorflow.NewCheckpointReader(self.weights)
        var_to_shape_map = reader.get_variable_to_shape_map()
        glb_var = tf.global_variables()
        with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:
            for key in var_to_shape_map:
                try:
                    var = tf.get_variable(key, trainable=False)
                    sess.run(var.assign(reader.get_tensor(key)))
                    print "    Assign pretrain model: " + key
                except ValueError:
                    print "    Ignore variable:" + key

        cubic_cls_score = tf.nn.softmax(self.result)
        timer = Timer()
        vispy_init()
        res =[]
        loop_parameters = np.arange(0,360,2)
        for data_idx in loop_parameters:  # DO NOT EDIT the "training_series",for the latter shuffle
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            feed_dict = self.cubic_rpn_grid(30, box_idx=0,
                                            angel=data_idx,
                                            scalar=1.0,#float(data_idx)/180.*1.0,
                                            translation=[0, 0,0])
            timer.tic()
            cubic_cls_score_ = sess.run(cubic_cls_score, feed_dict=feed_dict, options=run_options,run_metadata=run_metadata)
            timer.toc()
            cubic_cls_score_ = np.array(cubic_cls_score_)
            cubic_result = cubic_cls_score_.argmax(axis=1)
            res.append(cubic_cls_score_[0,1])
            # print 'rotation: {:3d}  score: {:>8,.7f} {:>8,.7f}  result: {}'.format(data_idx,cubic_cls_score_[0,0],cubic_cls_score_[0,1],cubic_result[0])

        plt.plot(loop_parameters, res)
        plt.grid(True, color='black', linestyle='--', linewidth='1')
        plt.title('Rubust Test')
        plt.xlabel('rotated angle metric:degree')
        plt.ylabel('score')
        plt.legend(['positive'])
        plt.savefig('Rotation.png')
        plt.show()

    def cubic_rpn_grid(self, data_idx, box_idx, angel, scalar, translation):
        blobs = self.dataset.get_minibatch(data_idx)  # get one batch
        lidarPoints = blobs['lidar3d_data']
        # rpnBoxes = blobs['gt_boxes_3d'][box_idx]
        rpnBoxes=np.array([10.832,2.4,-0.6,4,4,2,0.9])
        # rpnBoxes = np.array([17.832, -3.65, -0.3726, 4, 4, 2, 0.9])
        res = []
        display_stack = []
        if DEBUG:
            pass
            display_stack.append(pcd_vispy(lidarPoints, boxes=rpnBoxes, visible=False, multi_vis=True))

        for iidx, box in enumerate([rpnBoxes, rpnBoxes]):
            rpn_points, min_vertex, ctr_vertex = bounding_filter(lidarPoints, box)
            points_mv_ctr = np.subtract(rpn_points, ctr_vertex)  # using as feature
            # angel = random.rand()*np.pi*2 #[ 0~360]
            # scalar = 1.2 - random.rand()*0.4
            # translation = np.random.rand(3, 1) * 0.5
            angel = angel * 0.017453292  # angle to radius
            translation = np.array(translation, dtype=np.float32).reshape(3, 1)
            points_mv_ctr_rot_nobound = rot_sca_pc(points_mv_ctr, angel, scalar, translation)
            points_mv_ctr_rot, min_p, ctr_p = bounding_filter(points_mv_ctr_rot_nobound, [0, 0, 0])

            x_cub = np.divide(points_mv_ctr_rot[:, 0] - min_p[0], cfg.CUBIC_RES[0]).astype(np.int32)
            y_cub = np.divide(points_mv_ctr_rot[:, 1] - min_p[1], cfg.CUBIC_RES[1]).astype(np.int32)
            z_cub = np.divide(points_mv_ctr_rot[:, 2] - min_p[2], cfg.CUBIC_RES[2]).astype(np.int32)
            feature = np.hstack((np.ones([len(points_mv_ctr_rot[:, 3]), 1]),points_mv_ctr_rot[:, 3].reshape(-1, 1)))  # points_mv_ctr_rot

            cubic_feature = np.zeros(shape=cubic_size, dtype=np.float32)
            cubic_feature[
                x_cub, y_cub, z_cub] = feature  # TODO:select&add feature # points_mv_ctr  # using center coordinate system
            res.append(cubic_feature)

            if DEBUG and iidx == 0:
                box_mv = [box[0] - box[0], box[1] - box[1], box[2] - box[2], shape(0), shape(1), shape(2), 1, 1, 1]
                box_gt_mv = [box[0] - box[0], box[1] - box[1], box[2] - box[2], cfg.ANCHOR[0], cfg.ANCHOR[1],
                             cfg.ANCHOR[2], 1, 1, 1]
                show_feature = np.hstack((x_cub.reshape(-1, 1) - (shape(0) / 2), y_cub.reshape(-1, 1) - (shape(1) / 2),
                                          z_cub.reshape(-1, 1) - (shape(2) / 2), points_mv_ctr_rot[:, 3].reshape(-1, 1)))  # points_mv_ctr_rot
                cubic_show_feature = np.zeros(shape=cubic_show_size, dtype=np.float32)
                cubic_show_feature[
                    x_cub, y_cub, z_cub] = show_feature  # TODO:select&add feature # points_mv_ctr  # using center coordinate system
                display_stack.append(
                    pcd_vispy(cubic_show_feature.reshape(-1, 4), name='grid_' + str(iidx), boxes=np.array(box_mv),
                              visible=False, point_size=0.04, multi_vis=True))
                display_stack.append(
                    pcd_vispy(points_mv_ctr.reshape(-1, 4), name='origin_' + str(iidx), boxes=np.array(box_gt_mv),
                              visible=False, point_size=0.04, multi_vis=True))
        if DEBUG:
            pcd_show_now()
        stack_size = np.concatenate((np.array([-1]), cubic_size))
        return {self.cube_batch:np.array(res, dtype=np.float32).reshape(stack_size)}

def bounding_filter(points, box):
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

    return filter_points, np.array([x_min, y_min, z_min, 0.], dtype=np.float32), np.array([box[0], box[1], box[2], 0.],
                                                                                          dtype=np.float32)
def rot_sca_pc(points, rotation, scalar, translation):
    # points: numpy array;translation: moving scalar which should be small
    R = np.array([[np.cos(rotation), -np.sin(rotation), 0.],
                  [np.sin(rotation), np.cos(rotation), 0.],
                  [0, 0, 1]], dtype=np.float32)
    assert translation.shape == (3, 1), 'File rpn_3dcnn Function rot_sca_pc :T is  incompatible with transform'
    # T = np.random.randn(3, 1) * translation
    points_rot = np.matmul(R, points[:, 0:3].transpose()) + translation
    points_rot_sca = points_rot * scalar
    return np.hstack((points_rot_sca.transpose(), points[:, 3:]))

def network_training():
    net = CubicNet_Train()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        net.training(sess)


if __name__ == '__main__':
    network_training()
