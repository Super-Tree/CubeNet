# coding=utf-8
import _init_paths
from numpy import random
import os
from os.path import join as path_add
import math
import numpy as np
import tensorflow as tf
from tools.timer import Timer
from network.config import cfg
from tensorflow.python.client import timeline
from tools.data_visualize import vispy_init
from tools.printer import red, blue, yellow, cyan, green, purple, darkyellow
from easydict import EasyDict as edict
from boxes_factory import box_np_view
import multiprocessing
import cv2
from contextlib import contextmanager
from tools.utils import fast_hist
from tensorflow.python.ops import init_ops
import socket

DEBUG = False


class data_load(object):
    """
    valid negative  46260  positive  5940
    train negative 239576  positive 30924
    """

    def __init__(self, path, arg_, one_piece=False):
        self.path = path
        self.arg = arg_
        self.train_positive_cube_cnt = 20384  # TODO:better to check the number in every time use functions
        self.train_negative_cube_cnt = 239576
        self.valid_positive_cube_cnt = 3941
        self.valid_negative_cube_cnt = 46260

        self.TrainSet_POS = []
        self.TrainSet_NEG = []
        self.ValidSet_POS = []
        self.ValidSet_NEG = []

        self.dataset_TrainP_record = 0
        self.dataset_ValidP_record = 0
        self.dataset_TrainN_record = 0
        self.dataset_ValidN_record = 0

        self.conv3d_int_weights = 0
        if one_piece:
            self.eat_data_in_one_piece()
            self.load_all_data = True
        else:
            print(
                darkyellow('The data[TP({}) TN({}) VP({}) VN({})] will be eaten one by one while training ... '.format(
                    self.train_positive_cube_cnt, self.train_negative_cube_cnt, self.valid_positive_cube_cnt,
                    self.valid_negative_cube_cnt)))
            self.load_all_data = False

    def get_minibatch(self, idx_array, data_type='train', classify='positive'):
        one_piece = self.load_all_data
        if one_piece:
            if data_type == 'train' and classify == 'positive':
                extractor = self.TrainSet_POS
            elif data_type == 'train' and classify == 'negative':
                extractor = self.TrainSet_NEG
            elif data_type == 'valid' and classify == 'positive':
                extractor = self.ValidSet_POS
            else:
                extractor = self.ValidSet_NEG
            ret = extractor[idx_array].reshape(-1, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 1)
        else:
            if data_type == 'train':
                file_prefix = path_add(self.path, 'KITTI_TRAIN_BOX')
            else:
                file_prefix = path_add(self.path, 'KITTI_VALID_BOX')

            if classify == 'positive':
                file_prefix = path_add(file_prefix, 'POSITIVE')
            else:
                file_prefix = path_add(file_prefix, 'NEGATIVE')

            res = []
            for idx in idx_array:
                data = np.load(path_add(file_prefix, str(idx).zfill(6) + '.npy'))
                res.append(data)
            ret = np.array(res, dtype=np.uint8).reshape(-1, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 1)
        return ret

    def eat_data_in_one_piece(self):
        if not os.path.exists(path_add(self.path, 'data_in_one_piece')):
            os.mkdir(path_add(self.path, 'data_in_one_piece'))
        TrainSet_POS_file_name = path_add(self.path, 'data_in_one_piece', 'TrainSet_POS.npy')
        TrainSet_NEG_file_name = path_add(self.path, 'data_in_one_piece', 'TrainSet_NEG.npy')
        ValidSet_POS_file_name = path_add(self.path, 'data_in_one_piece', 'ValidSet_POS.npy')
        ValidSet_NEG_file_name = path_add(self.path, 'data_in_one_piece', 'ValidSet_NEG.npy')

        if not os.path.exists(path_add(self.path, 'filter_data_in_one_piece')):
            os.mkdir(path_add(self.path, 'filter_data_in_one_piece'))
        info_file_name = path_add(self.path, 'filter_data_in_one_piece', 'information_about_files.npy')
        TrainSet_POS_filter_file_name = path_add(self.path, 'filter_data_in_one_piece', 'Filter_TrainSet_POS.npy')
        ValidSet_POS_filter_file_name = path_add(self.path, 'filter_data_in_one_piece', 'Filter_ValidSet_POS.npy')
        TrainSet_NEG_filter_file_name = path_add(self.path, 'filter_data_in_one_piece', 'Filter_TrainSet_NEG.npy')
        ValidSet_NEG_filter_file_name = path_add(self.path, 'filter_data_in_one_piece', 'Filter_ValidSet_NEG.npy')

        if os.path.exists(TrainSet_POS_filter_file_name) and os.path.exists(ValidSet_POS_filter_file_name) \
                and os.path.exists(TrainSet_NEG_filter_file_name) and os.path.exists(ValidSet_NEG_filter_file_name) \
                and os.path.exists(info_file_name):
            print('Eating filtered data(Points more than {}) from npy zip file in folder:filter_data_in_one_piece ...'
                    .format(darkyellow('['+str(np.load(info_file_name))+']')))
            self.TrainSet_POS = np.load(TrainSet_POS_filter_file_name)
            self.TrainSet_NEG = np.load(TrainSet_NEG_filter_file_name)
            self.ValidSet_POS = np.load(ValidSet_POS_filter_file_name)
            self.ValidSet_NEG = np.load(ValidSet_NEG_filter_file_name)

            self.train_positive_cube_cnt = self.TrainSet_POS.shape[0]
            self.train_negative_cube_cnt = self.TrainSet_NEG.shape[0]
            self.valid_positive_cube_cnt = self.ValidSet_POS.shape[0]
            self.valid_negative_cube_cnt = self.ValidSet_NEG.shape[0]

            print('  emmm,there are TP:{} TN:{} VP:{} VN:{} in my belly.'.format(
                purple(str(self.TrainSet_POS.shape[0])),purple(str(self.TrainSet_NEG.shape[0])),
                purple(str(self.ValidSet_POS.shape[0])),purple(str(self.ValidSet_NEG.shape[0])), ))

            return None

        if os.path.exists(TrainSet_POS_file_name) and os.path.exists(TrainSet_NEG_file_name) \
                and os.path.exists(ValidSet_POS_file_name) and os.path.exists(ValidSet_NEG_file_name):
            print(blue('Let`s eating exiting data !'))
            self.TrainSet_POS = np.load(TrainSet_POS_file_name)
            self.TrainSet_NEG = np.load(TrainSet_NEG_file_name)
            self.ValidSet_POS = np.load(ValidSet_POS_file_name)
            self.ValidSet_NEG = np.load(ValidSet_NEG_file_name)
        else:
            print(darkyellow('Let`s eating raw data !'))
            train_pos_name_list = sorted(os.listdir(path_add(self.path, 'KITTI_TRAIN_BOX', 'POSITIVE')))
            train_neg_name_list = sorted(os.listdir(path_add(self.path, 'KITTI_TRAIN_BOX', 'NEGATIVE')))
            valid_pos_name_list = sorted(os.listdir(path_add(self.path, 'KITTI_VALID_BOX', 'POSITIVE')))
            valid_neg_name_list = sorted(os.listdir(path_add(self.path, 'KITTI_VALID_BOX', 'NEGATIVE')))
            for name in train_pos_name_list:
                data = np.load(path_add(self.path, 'KITTI_TRAIN_BOX', 'POSITIVE') + '/' + name)
                self.TrainSet_POS.append(data)
            self.TrainSet_POS = np.array(self.TrainSet_POS, dtype=np.uint8)
            np.save(TrainSet_POS_file_name, self.TrainSet_POS)
            print('  Yummy!')

            for name in train_neg_name_list:
                data = np.load(path_add(self.path, 'KITTI_TRAIN_BOX', 'NEGATIVE') + '/' + name)
                self.TrainSet_NEG.append(data)
            self.TrainSet_NEG = np.array(self.TrainSet_NEG, dtype=np.uint8)
            np.save(TrainSet_NEG_file_name, self.TrainSet_NEG)

            print('  Take another piece!')

            for name in valid_pos_name_list:
                data = np.load(path_add(self.path, 'KITTI_VALID_BOX', 'POSITIVE') + '/' + name)
                self.ValidSet_POS.append(data)
            self.ValidSet_POS = np.array(self.ValidSet_POS, dtype=np.uint8)
            np.save(ValidSet_POS_file_name, self.ValidSet_POS)
            print('  One more!')

            for name in valid_neg_name_list:
                data = np.load(path_add(self.path, 'KITTI_VALID_BOX', 'NEGATIVE') + '/' + name)
                self.ValidSet_NEG.append(data)
            self.ValidSet_NEG = np.array(self.ValidSet_NEG, dtype=np.uint8)
            np.save(ValidSet_NEG_file_name, self.ValidSet_NEG)
            print('  I`m full ...')
            print('All data has been saved in zip npy file!')

        print('There are TP:{} TN:{} VP:{} VN:{} and has been successfully eaten!'.format(
            self.TrainSet_POS.shape[0], self.TrainSet_NEG.shape[0], self.ValidSet_POS.shape[0],
            self.ValidSet_NEG.shape[0]))

        print(darkyellow(
            'Filter the positive data which has less points({}) inside ... '.format(self.arg.positive_points_needed)))
        train_sum = np.array([self.TrainSet_POS[i].sum() for i in range(self.TrainSet_POS.shape[0])])
        keep_mask1 = np.where(train_sum > self.arg.positive_points_needed)
        self.TrainSet_POS = self.TrainSet_POS[keep_mask1]
        np.save(TrainSet_POS_filter_file_name, self.TrainSet_POS)

        valid_sum = np.array([self.ValidSet_POS[i].sum() for i in range(self.ValidSet_POS.shape[0])])
        keep_mask2 = np.where(valid_sum > self.arg.positive_points_needed)
        self.ValidSet_POS = self.ValidSet_POS[keep_mask2]
        np.save(ValidSet_POS_filter_file_name, self.ValidSet_POS)

        np.save(ValidSet_NEG_filter_file_name, self.ValidSet_NEG)
        np.save(TrainSet_NEG_filter_file_name, self.TrainSet_NEG)
        np.save(info_file_name, self.arg.positive_points_needed)

        self.train_positive_cube_cnt = self.TrainSet_POS.shape[0]
        self.train_negative_cube_cnt = self.TrainSet_NEG.shape[0]
        self.valid_positive_cube_cnt = self.ValidSet_POS.shape[0]
        self.valid_negative_cube_cnt = self.ValidSet_NEG.shape[0]

        print(green('Done! TrainPositive remain: {},ValidPositive remain: {} and has been saved').
              format(self.TrainSet_POS.shape[0], self.ValidSet_POS.shape[0], ))

    def sti_test(self, idx):
        from tools.pcd_py_method.py_pcd import point_cloud
        prefix = '/home/hexindong/DATASET/DATA_BOXES/STI_BOX/pcd_car/'
        pc = point_cloud.from_path(prefix + str(idx) + '.pcd')
        return pc


class net_build(object):
    def __init__(self, channel, training=True):
        self.cube_input = tf.placeholder(dtype=tf.float32,
                                         shape=[None, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 1])
        self.cube_label = tf.placeholder(dtype=tf.int32, shape=[None])
        self.channel = channel

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

        self.extractor_int = 0
        self.extractor_weighs_float = 0
        self.extractor_outs = 0
        self.conv1 = self.conv3d_1.trainable_weights
        self.conv2 = self.conv3d_2.trainable_weights
        self.conv3 = self.conv3d_3.trainable_weights
        self.fc1 = self.dense_1.trainable_weights
        self.fc2 = self.dense_2.trainable_weights

        self.cube_score = self.apply(self.cube_input)

    def apply(self, inputs):
        assert len(inputs.shape.as_list()) == 5, ' The data`s dimension of the network isnot 5!'

        self.extractor_outs = self.shape_extractor(inputs)

        out_conv3d_1 = self.conv3d_1.apply(self.extractor_outs)
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

        return res

    def shape_extractor(self, inputs):

        def converter_grad(op, grad):
            return grad * 25

        def converter_op(kernel_w):
            extractor_int = np.greater(kernel_w, 0.0).astype(np.float32)

            return extractor_int

        def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
            # Need to generate a unique name to avoid duplicates:
            rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

            tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
            g = tf.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name}):
                return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

        def tf_extractor(x, name=None):
            with tf.name_scope(name, "shape_extractor", [x]) as name:
                z = py_func(converter_op,
                            [x],
                            [tf.float32],
                            name=name,
                            grad=converter_grad)  # <-- here's the call to the gradient
                return z[0]

        with tf.variable_scope('ShapeExtractor', reuse=tf.AUTO_REUSE) as scope:
            self.extractor_weighs_float = tf.get_variable('extractor_float', shape=[3, 3, 3, 1, self.channel[0]],
                                                          initializer=init_ops.variance_scaling_initializer)
            self.extractor_int = tf_extractor(self.extractor_weighs_float, name='extractor_int')
            res = tf.nn.conv3d(inputs, self.extractor_int, strides=[1, 1, 1, 1, 1], padding='SAME',
                               name='shape_feature')
            out = tf.reshape(res, [-1, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], self.channel[0]])

        return out

    def load_weigths(self, data_path, session, saver):
        import numpy as np
        try:
            if data_path.endswith('.ckpt'):
                saver.restore(session, data_path)
            else:
                data_dict = np.load(data_path).item()
                for key in data_dict:
                    with tf.variable_scope(key, reuse=True):
                        for subkey in data_dict[key]:
                            try:
                                var = tf.get_variable(subkey)
                                session.run(var.assign(data_dict[key][subkey]))
                                print "assign pretrain model " + subkey + " to " + key
                            except ValueError:
                                print "ignore " + key
        except:
            from tensorflow.python import pywrap_tensorflow
            reader = pywrap_tensorflow.NewCheckpointReader(data_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:
                for key in var_to_shape_map:
                    try:
                        var = tf.get_variable(key, trainable=False)
                        session.run(var.assign(reader.get_tensor(key)))
                        print "    Assign pretrain model: " + key
                    except ValueError:
                        print "    Ignore variable:" + key


class cube_train(object):
    def __init__(self, arg_, dataset, network):
        self.arg = arg_
        self.dataset = dataset
        self.network = network

        self.random_folder = cfg.RANDOM_STR
        self.saver = None
        self.writer=None
        self.current_saver_path = None
        self.training_record_init()

    def training_record_init(self):
        current_process_name_path = os.path.join(cfg.ROOT_DIR, 'process_record', self.arg.task_name, self.random_folder)
        if not os.path.exists(current_process_name_path):
            os.makedirs(current_process_name_path)
        self.current_saver_path = current_process_name_path
        os.system('cp %s %s' % (os.path.join(cfg.ROOT_DIR, 'experiment', 'net_cube_special.py'),
                                current_process_name_path))  # bkp of model&args def
        os.system('cp %s %s' % (os.path.join(cfg.ROOT_DIR,'lib','network', 'config.py'), current_process_name_path))

        self.writer = tf.summary.FileWriter(current_process_name_path, sess.graph, max_queue=1000, flush_secs=1)
        self.saver = tf.train.Saver(max_to_keep=1000)
        print(cyan('System recorder assistant deploy.'))

    def snapshot(self, sess, epoch_cnt=0):
        filename = os.path.join(self.current_saver_path, 'CubeOnly_epoch_{:d}'.format(epoch_cnt) + '.ckpt')
        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

    def shuffle_series(self):
        random.shuffle(self.dataset.TrainSet_POS)
        random.shuffle(self.dataset.TrainSet_NEG)
        random.shuffle(self.dataset.ValidSet_POS)
        random.shuffle(self.dataset.ValidSet_NEG)
        print('Shuffle the data series')

    @contextmanager
    def printoptions(self, *args, **kwargs):
        original_options = np.get_printoptions()
        np.set_printoptions(*args, **kwargs)
        try:
            yield
        finally:
            np.set_printoptions(**original_options)

    @staticmethod
    def cv2_rotation_trans(dict_share, key, data, center, angle, scale, translation):
        def cv2_op(image):
            image = image.reshape(cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], -1)
            M = cv2.getRotationMatrix2D(center, angle * 57.296, scale)
            rotated = cv2.warpAffine(image, M, (cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1]))
            # watch1 = image.sum(axis=-1, ).reshape(30, 30, 1)
            # watch2 = rotated.sum(axis=-1,).reshape(30,30,1)
            # cv2.namedWindow('a1',cv2.WINDOW_NORMAL)
            # cv2.namedWindow('a2',cv2.WINDOW_NORMAL)
            # cv2.imshow('a1', watch1)
            # cv2.imshow('a2',watch2)
            # cv2.waitKey()
            return rotated

        dict_share[key] = np.array(map(cv2_op, data), dtype=np.uint8).astype(np.float32)

    @staticmethod
    def single_process_task(share_dict, key, points_min, center, rotation, scalar, translation):
        # points: numpy array;  rotation: radius;
        points_ctr = np.subtract(points_min, center)
        R = np.array([[np.cos(rotation), -np.sin(rotation), 0.],
                      [np.sin(rotation), np.cos(rotation), 0.],
                      [0, 0, 1]], dtype=np.float32)
        assert points_ctr.shape[1] == 3, 'Points shape should be (n,3) not ({})'.format(points_ctr.shape)
        assert translation.shape == (3, 1), 'Translation: T is incompatible with transform'
        points_rot = np.matmul(R, points_ctr.transpose()) + translation
        points_rot_sca = points_rot * scalar

        share_dict[key] = points_rot_sca.transpose()

    def cube_augmentation(self, cube_array, aug_data=True, DEBUG=False):
        processor_cnt = self.arg.multi_process
        batch_size = cube_array.shape[0]
        assert batch_size % processor_cnt == 0, 'BatchSize must be multiples of {}!'.format(processor_cnt)
        delta = batch_size / processor_cnt
        mgr = multiprocessing.Manager()
        result_dict = mgr.dict()
        data_list = []
        processor_list = []
        angel_out = []
        for i in range(processor_cnt):
            if aug_data:
                angel = (random.rand() - 0.500) * np.pi * 0.9
                scalar = 1.05 - random.rand() * 0.1
                translation = np.random.rand(3, 1) * 0.08
            else:
                angel = 0.0
                scalar = 1.0
                translation = np.zeros([3, 1], dtype=np.float32)
            angel_out.append(angel)
            points_min = cube_array[delta * i:delta * i + delta]
            center = (cfg.CUBIC_SIZE[0] / 2, cfg.CUBIC_SIZE[1] / 2)
            # processor = multiprocessing.Process(target=cube_train.single_process_task,
            #                                     args=(result_dict, i, points_min, center, angel, scalar, translation))
            processor = multiprocessing.Process(target=cube_train.cv2_rotation_trans,
                                                args=(result_dict, i, points_min, center, angel, scalar, translation))
            processor_list.append(processor)

        map(multiprocessing.Process.start, processor_list)
        map(multiprocessing.Process.join, processor_list)
        for i in range(processor_cnt):
            data_list.append(result_dict[i])
        stack_cube = np.array(data_list).reshape(-1, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 1)

        return stack_cube

    def train_series_Gen(self, BatchSize, Type='train'):
        if Type == 'train':
            if (self.dataset.dataset_TrainP_record + BatchSize) < self.dataset.train_positive_cube_cnt:
                PositiveSet = range(self.dataset.dataset_TrainP_record, self.dataset.dataset_TrainP_record + BatchSize)
                self.dataset.dataset_TrainP_record += BatchSize
            else:
                breakpoint = self.dataset.dataset_TrainP_record + BatchSize - self.dataset.train_positive_cube_cnt
                tmp1 = range(self.dataset.dataset_TrainP_record, self.dataset.train_positive_cube_cnt)
                tmp2 = range(0, breakpoint)
                PositiveSet = tmp1 + tmp2
                self.dataset.dataset_TrainP_record = breakpoint

            if (self.dataset.dataset_TrainN_record + BatchSize) < self.dataset.train_negative_cube_cnt:
                NegativeSet = range(self.dataset.dataset_TrainN_record, self.dataset.dataset_TrainN_record + BatchSize)
                self.dataset.dataset_TrainN_record += BatchSize
            else:
                breakpoint2 = self.dataset.dataset_TrainN_record + BatchSize - self.dataset.train_negative_cube_cnt
                tmp3 = range(self.dataset.dataset_TrainN_record, self.dataset.train_negative_cube_cnt)
                tmp4 = range(0, breakpoint2)
                NegativeSet = tmp3 + tmp4
                self.dataset.dataset_TrainN_record = breakpoint2
            return PositiveSet, NegativeSet
        else:
            if (self.dataset.dataset_ValidP_record + BatchSize) < self.dataset.valid_positive_cube_cnt:
                PositiveSet = range(self.dataset.dataset_ValidP_record, self.dataset.dataset_ValidP_record + BatchSize)
                self.dataset.dataset_ValidP_record += BatchSize
            else:
                breakpoint = self.dataset.dataset_ValidP_record + BatchSize - self.dataset.valid_positive_cube_cnt
                tmp1 = range(self.dataset.dataset_ValidP_record, self.dataset.valid_positive_cube_cnt)
                tmp2 = range(0, breakpoint)
                PositiveSet = tmp1 + tmp2
                self.dataset.dataset_ValidP_record = breakpoint

            if (self.dataset.dataset_ValidN_record + BatchSize) < self.dataset.valid_negative_cube_cnt:
                NegativeSet = range(self.dataset.dataset_ValidN_record, self.dataset.dataset_ValidN_record + BatchSize)
                self.dataset.dataset_ValidN_record += BatchSize
            else:
                breakpoint2 = self.dataset.dataset_ValidN_record + BatchSize - self.dataset.valid_negative_cube_cnt
                tmp3 = range(self.dataset.dataset_ValidN_record, self.dataset.valid_negative_cube_cnt)
                tmp4 = range(0, breakpoint2)
                NegativeSet = tmp3 + tmp4
                self.dataset.dataset_ValidN_record = breakpoint2

            return PositiveSet, NegativeSet

    def training(self, sess):
        with tf.name_scope('loss_cube'):
            cube_score = self.network.cube_score
            cube_label = self.network.cube_label

            if self.arg.focal_loss:
                alpha = [1.0, 1.0]
                gamma = 2
                cube_probi = tf.nn.softmax(cube_score)
                tmp = tf.one_hot(cube_label, depth=2) * ((1 - cube_probi) ** gamma) * tf.log(
                    [cfg.EPS, cfg.EPS] + cube_probi) * alpha
                cube_cross_entropy = tf.reduce_mean(-tf.reduce_sum(tmp, axis=1))
            else:
                cube_probi = tf.nn.softmax(cube_score)  # use for debug
                tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cube_score, labels=cube_label)
                cube_cross_entropy = tf.reduce_mean(tmp)

            loss = cube_cross_entropy

        with tf.name_scope('train_op'):
            global_step = tf.Variable(1, trainable=False, name='Global_Step')
            lr = tf.train.exponential_decay(self.arg.lr, global_step, 1000, 0.90, name='decay-Lr')
            train_op = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(loss, global_step=global_step)

        with tf.name_scope('train_cubic'):
            extractor_int = self.network.extractor_int
            extractor_float = self.network.extractor_weighs_float

            # extractor_F_grad = tf.gradients(loss, extractor_float)
            # extractor_Int_grad = tf.gradients(loss, extractor_int)
            # conv1_grad = tf.gradients(loss, self.network.conv1)
            # conv2_grad = tf.gradients(loss, self.network.conv2)
            # conv3_grad = tf.gradients(loss, self.network.conv3)
            # fc1_grad = tf.gradients(loss, self.network.fc1)
            # fc2_grad = tf.gradients(loss, self.network.fc2)

            tf.summary.image('extractor_int', tf.reshape(extractor_int, [-1, 27, 32, 1]))
            # tf.summary.image('extractor_float', tf.reshape(extractor_float, [-1, 27, 32, 1]))
            # tf.summary.image('conv1_kernel', tf.reshape(self.network.conv1[0], [-1, 27, 32, 1]), max_outputs=3)
            # tf.summary.image('conv2_kernel', tf.reshape(self.network.conv2[0], [-1, 27, 64, 1]), max_outputs=3)
            # tf.summary.image('conv3_kernel', tf.reshape(self.network.conv3[0], [-1, 27, 128, 1]), max_outputs=3)
            #
            # tf.summary.histogram('float_grad', extractor_F_grad)
            # tf.summary.histogram('Int_grad', extractor_Int_grad)
            # tf.summary.histogram('conv1_grad', conv1_grad[0])
            # tf.summary.histogram('conv2_grad', conv2_grad[0])
            # tf.summary.histogram('conv3_grad', conv3_grad[0])
            # tf.summary.histogram('fc1_grad', fc1_grad[0])
            # tf.summary.histogram('fc2_grad', fc2_grad[0])

            tf.summary.scalar('total_loss', loss)
            glb_var = tf.global_variables()
            # for var in glb_var:
            # tf.summary.histogram(var.name, var)
            merged_op = tf.summary.merge_all()

        with tf.name_scope('valid_cubic'):
            epoch_cubic_recall = tf.placeholder(dtype=tf.float32)
            cubic_recall_smy_op = tf.summary.scalar('cubic_recall', epoch_cubic_recall)
            epoch_cubic_precise = tf.placeholder(dtype=tf.float32)
            cubic_precise_smy_op = tf.summary.scalar('cubic_precise', epoch_cubic_precise)

            epoch_extractor_occupy = tf.placeholder(dtype=tf.float32)
            cubic_occupy_smy_op = tf.summary.scalar('extractor_occupy', epoch_extractor_occupy)

            valid_summary_op = tf.summary.merge([cubic_recall_smy_op, cubic_precise_smy_op, cubic_occupy_smy_op])

        with tf.name_scope('load_weights'):
            sess.run(tf.global_variables_initializer())
            if self.arg.weights is not None:
                self.network.load_weigths(self.arg.weights, sess, self.saver)
                print 'Loading pre-trained model weights from {:s}'.format(red(self.arg.weights))
            else:
                print 'The network will be {} from default initialization!'.format(yellow('re-trained'))
        timer = Timer()
        if DEBUG:
            pass
            vispy_init()
        cube_label_gt = np.concatenate((np.ones([self.arg.batch_size]), np.zeros([self.arg.batch_size]))).astype(
            np.int32)
        train_epoch_cnt = int(self.dataset.train_positive_cube_cnt / self.arg.batch_size / 2)
        training_series = range(8)#range(train_epoch_cnt)  # train_epoch_cnt
        for epo_cnt in range(self.arg.epoch_iters):
            for data_idx in training_series:
                iter = global_step.eval()
                timer.tic()
                series = self.train_series_Gen(self.arg.batch_size, 'train')
                data_batchP = self.dataset.get_minibatch(series[0], data_type='train', classify='positive')
                data_batchN = self.dataset.get_minibatch(series[1], data_type='train', classify='negative')
                data_batch = np.vstack((data_batchP, data_batchN))
                timer.toc()
                time1 = timer.average_time

                timer.tic()
                data_aug = self.cube_augmentation(data_batch, aug_data=True, DEBUG=False)
                timer.toc()
                time2 = timer.average_time
                if DEBUG:
                    a = data_batch[data_idx].sum()
                    b = data_batch[data_idx].sum()
                    if a != b:
                        print 'There is some points loss'
                    else:
                        print 'points cnt: ', a
                    box_np_view(data_aug[data_idx], data_aug[data_idx + self.arg.batch_size])
                feed_dict = {self.network.cube_input: data_aug,
                             self.network.cube_label: cube_label_gt,
                             }
                timer.tic()
                extractor_int_, cube_probi_, cube_label_, loss_, merge_op_, _ = \
                    sess.run([extractor_int, cube_probi, cube_label, loss, merged_op,
                              train_op], feed_dict=feed_dict)
                timer.toc()

                if iter % 4 == 0:
                    predict_result = cube_probi_.argmax(axis=1)
                    one_train_hist = fast_hist(cube_label_gt, predict_result)
                    occupy_part = extractor_int_.sum() / extractor_int_.size
                    print 'Training step: {:3d} loss: {:.4f} occupy: {}%({}) inference_time: {:.3f} '.format(iter,
                                                                                                             loss_, int(
                            occupy_part * 100), extractor_int_.sum(), timer.average_time)
                    # print('    class bg precision = {:.3f}  recall = {:.3f}'.format(
                    #     (one_train_hist[0, 0] / (one_train_hist[0, 0] + one_train_hist[1, 0] + 1e-6)),
                    #     (one_train_hist[0, 0] / (one_train_hist[0, 0] + one_train_hist[0, 1] + 1e-6))))
                    print '    class car precision = {:.3f}  recall = {:.3f}'.format(
                        (one_train_hist[1, 1] / (one_train_hist[1, 1] + one_train_hist[0, 1] + 1e-6)),
                        (one_train_hist[1, 1] / (one_train_hist[1, 1] + one_train_hist[1, 0] + 1e-6))), '\n'
                    if socket.gethostname() == "hexindong" and False:
                        with self.printoptions(precision=2, suppress=False, linewidth=10000):
                            print 'scores: {}'.format(cube_probi_[:, 1])
                            print 'divine:', str(predict_result)
                            print 'labels:', str(cube_label_), '\n'

                if iter % 2 == 0 and cfg.TRAIN.TENSORBOARD:
                    pass
                    self.writer.add_summary(merge_op_, iter)
                if (iter % 3000 == 0 and cfg.TRAIN.DEBUG_TIMELINE) or iter == 200:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _ = sess.run([cube_score], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    # chrome://tracing
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(cfg.LOG_DIR + '/' + 'training-step-' + str(iter).zfill(7) + '.ctf.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                    trace_file.close()

            if epo_cnt % 2 == 0 and cfg.TRAIN.EPOCH_MODEL_SAVE:
                pass
                self.snapshot(sess, epo_cnt)
            if cfg.TRAIN.USE_VALID:
                with tf.name_scope('valid_cubic_' + str(epo_cnt + 1)):
                    print 'Valid the net at the end of epoch_{} ...'.format(epo_cnt + 1)
                    hist = np.zeros((cfg.NUM_CLASS, cfg.NUM_CLASS), dtype=np.float32)
                    valid_epoch_cnt = int(self.dataset.valid_positive_cube_cnt / self.arg.batch_size / 2)
                    for data_idx in range(valid_epoch_cnt):
                        series = self.train_series_Gen(self.arg.batch_size, 'valid')
                        data_batchP = self.dataset.get_minibatch(series[0], data_type='valid', classify='positive')
                        data_batchN = self.dataset.get_minibatch(series[1], data_type='valid', classify='negative')
                        data_batch = np.vstack((data_batchP, data_batchN))

                        feed_dict_ = {self.network.cube_input: data_batch,
                                      self.network.cube_label: cube_label_gt,
                                      }
                        valid_cls_score_ = sess.run(cube_score, feed_dict=feed_dict_)

                        valid_result = valid_cls_score_.argmax(axis=1)
                        one_hist = fast_hist(cube_label_gt, valid_result)
                        hist += one_hist
                        if cfg.TRAIN.VISUAL_VALID:
                            print 'Valid step: {:d}/{:d}'.format(data_idx + 1, valid_epoch_cnt)
                            print('    class bg precision = {:.3f}  recall = {:.3f}'.format(
                                (one_hist[0, 0] / (one_hist[0, 0] + one_hist[1, 0] + 1e-6)),
                                (one_hist[0, 0] / (one_hist[0, 0] + one_hist[0, 1] + 1e-6))))
                            print('    class car precision = {:.3f}  recall = {:.3f}'.format(
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[0, 1] + 1e-6)),
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[1, 0] + 1e-6))))
                        if data_idx % 20 == 0 and cfg.TRAIN.TENSORBOARD:
                            pass
                            # train_writer.add_summary(valid_result_, data_idx/20+epo_cnt*1000)
                valid_extractor_int_ = sess.run(extractor_int)
                extractor_occupy = valid_extractor_int_.sum() / valid_extractor_int_.size
                precise_total = hist[1, 1] / (hist[1, 1] + hist[0, 1] + 1e-6)
                recall_total = hist[1, 1] / (hist[1, 1] + hist[1, 0] + 1e-6)
                valid_res = sess.run(valid_summary_op, feed_dict={epoch_cubic_recall: recall_total,
                                                                  epoch_cubic_precise: precise_total,
                                                                  epoch_extractor_occupy: extractor_occupy
                                                                  })
                self.writer.add_summary(valid_res, epo_cnt + 1)
                print 'Validation of epoch_{}: cubic_precision = {:.3f}  cubic_recall = {:.3f}' \
                    .format(epo_cnt + 1, precise_total, recall_total)
            self.shuffle_series()
        print yellow('Training process has done, enjoy every day !')


if __name__ == '__main__':
    arg = edict()
    arg.lr = 0.02
    arg.imdb_type = 'kitti'
    arg.use_demo = True
    arg.weights = None  # '/home/hexindong/Videos/cubic-local/MODEL_weights/tmp/CubeOnly_epoch_928.ckpt'
    arg.focal_loss = True
    arg.use_aug_data_method = True
    arg.positive_points_needed = 40
    arg.epoch_iters = 1000
    arg.one_piece = True
    arg.task_name = 'cube_2state_A0'

    if socket.gethostname() == "szstdzcp0325":
        arg.batch_size = 80
        arg.multi_process = 4
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        arg.batch_size = 340
        arg.multi_process = 20

    data_path = '/home/likewise-open/SENSETIME/hexindong/ProjectDL/cubic-local/DATASET/KITTI/object/box_car_only'
    DataSet = data_load(data_path, arg, one_piece=arg.one_piece)

    NetWork = net_build([32, 64, 128, 128, 64, 2])

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        task = cube_train(arg, DataSet, NetWork)
        task.training(sess)

