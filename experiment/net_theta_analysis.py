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
from tools.data_visualize import boxary2dic
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from multiprocessing import Process,Queue
MSG_QUEUE = Queue(200)


BATCH_CNT = 2
shape = lambda i: int(np.ceil(np.round(cfg.ANCHOR[i] / cfg.CUBIC_RES[i], 3)))  # Be careful about python number  decimal
cubic_size = [shape(0), shape(1), shape(2), 2]
cubic_show_size = [shape(0), shape(1), shape(2), 4]
cubic_batch_size = [BATCH_CNT, shape(0), shape(1), shape(2), 2]

class cubic(object):
    def __init__(self,input):
        # T:[B,30,30,15,2] ->[B,30,30]
        bi_bv = tf.reduce_max(input[:, :, :, :, 0], axis=3, keep_dims=True)
        layer = tf.reshape(bi_bv, [BATCH_CNT, 30, 30, 1])

        self.input_sum_op = tf.summary.image(name="inout",tensor=layer,max_outputs=1)
        self.img_tf = layer[0]
        layer = tf.layers.conv2d(layer, filters=32, kernel_size=5, strides=[1, 1], padding="same",
                                 activation=tf.nn.relu, name=None, use_bias=False)

        for i in range(32):
             tf.summary.image(name="conv1_sum_op",tensor=layer[:,:,:,i:i+1],max_outputs=1)
        # layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.max_pooling2d(layer, [2, 2], [2, 2])
        for i in range(32):
            tf.summary.image(name="pool1_sum_op",tensor=layer[:,:,:,i:i+1])
        layer = tf.layers.conv2d(layer, filters=64, kernel_size=5, strides=[1, 1], padding="same",
                                 activation=tf.nn.relu, name=None, use_bias=True)
        for i in range(64):
            tf.summary.image(name="conv2_sum_op",tensor=layer[:,:,:,i:i+1],max_outputs=1)
        # layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.max_pooling2d(layer, [2, 2], [2, 2])
        for i in range(64):
            tf.summary.image(name="pool2_sum_op",tensor=layer[:,:,:, i:i+1],max_outputs=1)
        # layer = tf.layers.average_pooling2d(layer,[6,6],[6,6])
        layer = tf.reshape(layer, [BATCH_CNT, -1])
        # layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.dense(layer, 1024, use_bias=True)
        # layer = tf.nn.dropout(layer,0.4)  # TODO: test remove
        layer = tf.nn.relu(layer)
        layer = tf.layers.dense(layer, 1, use_bias=True)
        layer = tf.reshape(layer, [-1])
        # layer = tf.layers.conv2d(layer,filters=32,kernel_size=3,strides=[1, 1],activation=tf.nn.relu,name=None)
        self.res = layer
class CubicNet_Train(object):
    def __init__(self):
        self.weights = '/home/hexindong/ws_dl/pyProj/cubic-local/MODEL_weights/THETA_1/weights/CubicNet_iter_345854.ckpt'
        arg = edict()
        arg.imdb_type = 'kitti'
        arg.use_demo = True
        self.dataset = dataset_KITTI_train(arg)
        self.index = tf.placeholder(tf.float32, shape=[1], name='index')
        self.cube = tf.placeholder(tf.float32, shape=[None,30,30,15,2], name='cubes')
        with tf.variable_scope('cubic_theta', reuse=tf.AUTO_REUSE) as scope:
            self.cubic_theta = cubic(self.cube)

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

        timer = Timer()
        vispy_init()
        res =[]
        input_series=[]
        merge_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(cfg.LOG_DIR, sess.graph, max_queue=1000,flush_secs=1)
        loop_parameters = np.arange(-90,90,1)
        data_id = 1
        box_cnt = 0
        for data_idx in loop_parameters:  # DO NOT EDIT the "training_series",for the latter shuffle
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            debug_mod = True if data_idx==0 else False
            # debug_mod = True
            feed_dict = self.cubic_rpn_grid(data_id,box_idx=box_cnt,
                                            angel=data_idx,
                                            scalar=1.00,#float(data_idx)/180.*1.0,
                                            translation=[0,0,0],DEBUG=debug_mod)

            timer.tic()
            img_tf_,cubic_theta_,merge_op_ = sess.run([self.cubic_theta.img_tf, self.cubic_theta.res, merge_op], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            timer.toc()
            input_series.append(img_tf_)
            res.append(cubic_theta_[0]*180/3.1415926)
            # print 'rotation: {:3d}  score: {:>8,.7f} {:>8,.7f}  result: {}'.format(data_idx,cubic_cls_score_[0,0],cubic_cls_score_[0,1],cubic_result[0])
            train_writer.add_summary(merge_op_,data_idx)
        imge_op = tf.summary.image("imagesss",np.array(input_series,dtype=np.float32).reshape(-1,30,30,1),max_outputs=180)
        imge_op_ = sess.run(imge_op)
        train_writer.add_summary(imge_op_, 1)
        plt.plot(loop_parameters, res)
        plt.grid(True, color='black', linestyle='--', linewidth='1')
        plt.title('Car_{}_{}'.format(data_id,box_cnt))
        plt.xlabel('gt_yaw+')
        plt.ylabel('pred-yaw')
        plt.legend(['positive'])
        plt.savefig('Roation_of_Car2.png')

        xmajorLocator = MultipleLocator(10)  # 将x主刻度标签设置为20的倍数
        xmajorFormatter = FormatStrFormatter('%1.0f')  # 设置x轴标签文本的格式
        xminorLocator = MultipleLocator(5)  # 将x轴次刻度标签设置为5的倍数

        ymajorLocator = MultipleLocator(10)  # 将y轴主刻度标签设置为0.5的倍数
        ymajorFormatter = FormatStrFormatter('%1.0f')  # 设置y轴标签文本的格式
        yminorLocator = MultipleLocator(5)  # 将此y轴次刻度标签设置为0.1的倍数

        ax = plt.axes()

        # 设置主刻度标签的位置,标签文本的格式
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_major_formatter(xmajorFormatter)

        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_major_formatter(ymajorFormatter)

        # 显示次刻度标签的位置,没有标签文本
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)

        ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
        ax.yaxis.grid(True, which='minor')  # y坐标轴的网格使用次刻度

        plt.show()

    def cubic_rpn_grid(self, data_idx, box_idx, angel, scalar, translation,DEBUG = False):
        blobs = self.dataset.get_minibatch(data_idx)  # get one batch
        lidarPoints = blobs['lidar3d_data']
        rpnBoxes_ = blobs['gt_boxes_3d'][box_idx]
        rpnBoxes=np.array([rpnBoxes_[0],rpnBoxes_[1],rpnBoxes_[2],4.,4.,2.,rpnBoxes_[6],rpnBoxes_[7],])
        box=dict({"center":rpnBoxes[0:3].reshape(1,3),"size":rpnBoxes[3:6].reshape(1,3),'cls_rpn':rpnBoxes[6].reshape(1,1)*4,"yaw":rpnBoxes[7].reshape(1,1),"score":np.ones([1,1],dtype=np.float32)})
        # rpnBoxes = np.array([17.832, -3.65, -0.3726, 4, 4, 2, 0.9])
        res = []
        display_stack = []
        if DEBUG:
            pass
            display_stack.append(pcd_vispy(lidarPoints, boxes=box, visible=False, multi_vis=True))

        for iidx, box in enumerate([rpnBoxes, rpnBoxes]):
            rpn_points, min_vertex, ctr_vertex = bounding_filter(lidarPoints, box)
            points_mv_ctr = np.subtract(rpn_points, ctr_vertex)  # using as feature
            # angel = random.rand()*np.pi*2 #[ 0~360]
            # scalar = 1.2 - random.rand()*0.4
            # translation = np.random.rand(3, 1) * 0.5
            angel = angel * 0.017453292 + rpnBoxes_[7]  # angle to radius
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
                box_mv = [box[0] - box[0], box[1] - box[1], box[2] - box[2], shape(0), shape(1), shape(2), 1, 0, 1]
                box_gt_mv = [box[0] - box[0], box[1] - box[1], box[2] - box[2], cfg.ANCHOR[0], cfg.ANCHOR[1],
                             cfg.ANCHOR[2], 1,0, 1]
                show_feature = np.hstack((x_cub.reshape(-1, 1) - (shape(0) / 2), y_cub.reshape(-1, 1) - (shape(1) / 2),
                                          z_cub.reshape(-1, 1) - (shape(2) / 2), points_mv_ctr_rot[:, 3].reshape(-1, 1)))  # points_mv_ctr_rot
                cubic_show_feature = np.zeros(shape=cubic_show_size, dtype=np.float32)
                cubic_show_feature[
                    x_cub, y_cub, z_cub] = show_feature  # TODO:select&add feature # points_mv_ctr  # using center coordinate system
                display_stack.append(
                    pcd_vispy(cubic_show_feature.reshape(-1, 4), name='grid_' + str(iidx), boxes=boxary2dic(np.array(box_mv).reshape(1,-1)),
                              visible=False, point_size=0.02, multi_vis=True))
                display_stack.append(
                    pcd_vispy(points_mv_ctr.reshape(-1, 4), name='origin_' + str(iidx), boxes=boxary2dic(np.array(box_gt_mv).reshape(1,-1)),
                              visible=False, point_size=0.02, multi_vis=True))
        if DEBUG:
            pcd_show_now()
        stack_size = np.concatenate((np.array([-1]), cubic_size))
        return {self.cube:np.array(res, dtype=np.float32).reshape(stack_size),
                self.index:np.array(angel,dtype=np.float32).reshape(1)}

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
