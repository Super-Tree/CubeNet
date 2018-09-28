# coding=utf-8
import _init_paths
from numpy import random
import random as py_random
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tools.timer import Timer
from network.config import cfg
from tensorflow.python.client import timeline
from tools.data_visualize import vispy_init
from tools.printer import red, blue, yellow, cyan, green, purple, darkyellow
from easydict import EasyDict as edict
from tools.data_visualize import pcd_show_now,pcd_vispy_standard
import multiprocessing
import cv2
from contextlib import contextmanager
from tools.utils import fast_hist
from tensorflow.python.ops import init_ops
import socket
import cPickle
from boxes_gt_factory import DataLoader
DEBUG = False

class cube_filter_data_load(object):
    """
    valid negative  46260  positive  5940
    train negative 239576  positive 30924
    """

    def __init__(self, path, arg_, one_piece=False):

        self.path = path
        self.arg = arg_

        self.dataset_TrainP_record = 0
        self.dataset_ValidP_record = 0
        self.dataset_TrainN_record = 0
        self.dataset_ValidN_record = 0

        self.negative_filtered_cube_file=os.path.join(self.path,'Filter_TrainSet_NEG.npy')
        self.positive_filtered_cube_file=os.path.join(self.path,'GT_Filtered_Cubes_Positive.pkl')
        self.positive_filtered_label_file=os.path.join(self.path,'GT_Cubes_Filtered_Positive_Label.pkl')
        self.data_info_file=os.path.join(self.path,'GT_Filtered_Cubes_Info.pkl')
        total_positive_filtered_cubes,total_negative_filtered_cubes,total_positive_filtered_labels =\
            self.eat_data_in_one_piece()
        self.positive_cube_length=len(total_positive_filtered_cubes)
        self.negative_cube_length=len(total_negative_filtered_cubes)

        self.training_positive_indice_file = os.path.join(self.path, 'training_positive_series.npy')
        self.validing_positive_indice_file = os.path.join(self.path, 'validing_positive_series.npy')
        self.training_negative_indice_file = os.path.join(self.path, 'training_negative_series.npy')
        self.validing_negative_indice_file = os.path.join(self.path, 'validing_negative_series.npy')
        self.training_pos_indice, self.validing_pos_indice,self.training_neg_indice, self.validing_neg_indice \
            = self.assign_partition_index(self.training_positive_indice_file, self.validing_positive_indice_file,
                                          self.training_negative_indice_file, self.validing_negative_indice_file)

        self.ValidSet_POS = total_positive_filtered_cubes[self.validing_pos_indice]
        self.ValidSet_NEG = total_negative_filtered_cubes[self.validing_neg_indice]
        self.TrainSet_POS = total_positive_filtered_cubes[self.training_pos_indice]
        self.TrainSet_NEG = total_negative_filtered_cubes[self.training_neg_indice]

        self.TrainSet_POS_LABEL = total_positive_filtered_labels[self.training_pos_indice]
        self.ValidSet_POS_LABEL = total_positive_filtered_labels[self.validing_pos_indice]

        self.train_positive_cube_cnt = len(self.TrainSet_POS)  # TODO:better to check the number in every time use functions
        self.train_negative_cube_cnt = len(self.TrainSet_NEG)
        self.valid_positive_cube_cnt = len(self.ValidSet_POS)
        self.valid_negative_cube_cnt = len(self.ValidSet_NEG)

        color=lambda x:darkyellow(str(x))
        print(('The data[TP({}) TN({}) VP({}) VN({})] has been loaded ... '.format(
            color(self.train_positive_cube_cnt), color(self.train_negative_cube_cnt), color(self.valid_positive_cube_cnt),
            color(self.valid_negative_cube_cnt))))

    def get_minibatch(self, idx_array, data_type='train', classify='positive'):
        if data_type == 'train' and classify == 'positive':
            extractor = self.TrainSet_POS
            label_extractor=self.TrainSet_POS_LABEL
        elif data_type == 'train' and classify == 'negative':
            extractor = self.TrainSet_NEG
        elif data_type == 'valid' and classify == 'positive':
            extractor = self.ValidSet_POS
            label_extractor = self.ValidSet_POS_LABEL
        else:
            extractor = self.ValidSet_NEG
        if classify=='negative':
            ret = extractor[idx_array].reshape(-1, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 1)
            return ret
        else:
            ret = extractor[idx_array].reshape(-1, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 1)
            return ret,label_extractor[idx_array].reshape(-1, 8)

    def eat_data_in_one_piece(self):

        if not(os.path.exists(self.positive_filtered_cube_file) and
               os.path.exists(self.negative_filtered_cube_file) and
               os.path.exists(self.positive_filtered_label_file) and
               os.path.exists(self.data_info_file)):
            print(red("The existing data is not complete"))
            print self.positive_filtered_cube_file,self.negative_filtered_cube_file,self.positive_filtered_label_file,self.data_info_file
            return 0

        print "Starting to load existing pkl filtered cubes ..."
        with open(self.positive_filtered_cube_file, 'rb') as fid:
            positive_filtered_cubes = cPickle.load(fid)

        negative_filtered_cubes = np.load(self.negative_filtered_cube_file)

        with open(self.positive_filtered_label_file, 'rb') as fid:
            positive_filtered_labels = cPickle.load(fid)
        with open(self.data_info_file, 'rb') as fid:
            data_info = cPickle.load(fid)
        print "   Total dataset[positive:{},negative{}],with cube info:{}".format(darkyellow(str(len(positive_filtered_cubes))),darkyellow(str(len(negative_filtered_cubes))), data_info)

        return positive_filtered_cubes,negative_filtered_cubes,positive_filtered_labels

    def assign_partition_index(self, training_indice_file, validing_indice_file,train_negative_file,valid_negative_file):

        if os.path.exists(self.training_positive_indice_file) and os.path.exists(self.validing_positive_indice_file) \
            and os.path.exists(self.training_negative_indice_file) and os.path.exists(self.validing_negative_indice_file):
            training_indice_np = np.load(training_indice_file)
            validing_indice_np = np.load(validing_indice_file)
            training_neg_indice_np = np.load(train_negative_file)
            validing_neg_indice_np = np.load(valid_negative_file)
        else:
            input_num = self.positive_cube_length
            validing_indice=[]
            training_indice = sorted(py_random.sample(range(input_num), int(input_num * 0.65)))

            for i in range(input_num):
                if i not in training_indice:
                    validing_indice.append(i)
            training_indice_np=np.array(training_indice)
            validing_indice_np=np.array(validing_indice)

            np.save(training_indice_file,training_indice_np)
            np.save(validing_indice_file,validing_indice_np)
            ##########################################################
            input_num = self.negative_cube_length
            validing_neg_indice=[]
            training_neg_indice = sorted(py_random.sample(range(input_num), int(input_num * 0.65)))

            for i in range(input_num):
                if i not in training_neg_indice:
                    validing_neg_indice.append(i)
            training_neg_indice_np=np.array(training_neg_indice)
            validing_neg_indice_np=np.array(validing_neg_indice)

            np.save(train_negative_file,training_neg_indice_np)
            np.save(valid_negative_file,validing_neg_indice_np)
            print("   Partition of dataset has been done and recoded")

        return training_indice_np,validing_indice_np,training_neg_indice_np,validing_neg_indice_np

class net_build(object):
    def __init__(self, channel,arg_, training=True):
        self.arg=arg_
        self.cube_input = tf.placeholder(dtype=tf.float32,shape=[None, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 1])

        self.cube_cls_label = tf.placeholder(dtype=tf.int32, shape=[None])
        self.cube_size_label = tf.placeholder(dtype=tf.float32, shape=[None,3])
        self.cube_ctr_label = tf.placeholder(dtype=tf.float32, shape=[None,3])
        self.cube_yaw_component_label = tf.placeholder(dtype=tf.float32, shape=[None,2])

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
            self.dense_2 = tf.layers.Dense(channel[5], _reuse=tf.AUTO_REUSE, _scope=scope,kernel_initializer=init_ops.variance_scaling_initializer)

        self.extractor_int = 0
        self.extractor_weighs_float = 0
        self.extractor_outs = 0
        self.conv1 = self.conv3d_1.trainable_weights
        self.conv2 = self.conv3d_2.trainable_weights
        self.conv3 = self.conv3d_3.trainable_weights
        self.fc1 = self.dense_1.trainable_weights
        self.fc2 = self.dense_2.trainable_weights

        self.outputs = self.apply(self.cube_input)

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
        amplifier_rate=self.arg.amplifier_rate

        def converter_grad(op, grad):
            return grad * amplifier_rate

        def converter_op(kernel_w):
            kernel_w_int = np.zeros_like(kernel_w, dtype=np.float32)
            extractor_int_pos = np.greater(kernel_w, 0.33).astype(np.float32)
            extractor_int_neg = np.less(kernel_w, -0.33).astype(np.float32) * -1.0

            return kernel_w_int + extractor_int_pos + extractor_int_neg

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
            self.extractor_weighs_float = tf.get_variable('extractor_float', shape=[5, 5, 5, 1, self.channel[0]],
                                                          initializer=tf.random_uniform_initializer(-1.,1.))
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

    def extractor_init(self):
        # shape = [3, 3, 3, 1, self.channel[0]
        # self.init_values=np.random.randn()
        pass

class cube_train(object):
    def __init__(self, arg_, dataset, network):
        self.arg = arg_
        self.ratio=self.arg.loss_weights
        self.dataset = dataset
        self.network = network

        self.random_folder = cfg.RANDOM_STR
        self.saver = None
        self.writer = None
        self.current_saver_path = None
        self.training_record_init()

    def training_record_init(self):
        current_process_name_path = os.path.join(cfg.ROOT_DIR, 'process_record', self.arg.task_name, self.random_folder)
        current_process_wights_path = os.path.join(current_process_name_path,"weights")
        current_runing_file_name=os.path.basename(sys.argv[0])
        if not os.path.exists(current_process_wights_path):
            os.makedirs(current_process_wights_path)
        self.current_saver_path = current_process_wights_path
        os.system('cp %s %s' % (os.path.join(cfg.ROOT_DIR, 'experiment', current_runing_file_name),
                                current_process_name_path))  # bkp of model&args def
        os.system('cp %s %s' % (os.path.join(cfg.ROOT_DIR, 'lib', 'network', 'config.py'), current_process_name_path))

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

        self.dataset.dataset_TrainP_record=0
        self.dataset.dataset_TrainN_record=0
        self.dataset.dataset_ValidP_record=0
        self.dataset.dataset_ValidN_record=0

        print('Shuffle the data series and reset dataset recorder!')

    def huber_loss(self,error, delta=3.0):
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = (abs_error - quadratic)
        losses = 0.5 * quadratic ** 2 + delta * linear
        return tf.reduce_mean(losses)

    def decompose_yaw(self,yaw_array):
        #//see the notebook page 1

        #make the theta in [-pi,_pi]
        indice1=np.where(yaw_array>np.pi)
        yaw_array[indice1] -= 2*np.pi
        indice2=np.where(yaw_array< -np.pi)
        yaw_array[indice2]+= 2*np.pi

        #make the theta in [-pi/2,_pi/2] #TODO:declaration:assuming that the car oriente forward
        indice1=np.where(yaw_array>np.pi/2)
        yaw_array[indice1] -= np.pi
        indice2=np.where(yaw_array< -np.pi/2)
        yaw_array[indice2]+= np.pi

        x_ary=np.cos(yaw_array).reshape(-1,1)
        y_ary=np.sin(yaw_array).reshape(-1,1)
        ret = np.hstack((x_ary, y_ary))
        # yaw = np.arctan2(y_ary,x_ary)
        # #TODO:hxd:todo: reduce the half angle
        # differ=yaw_array.reshape(-1)-yaw.reshape(-1)

        return ret

    def format_convertor(self,type,inputs):
        if type=='input':
            pcs_np=inputs[0]
            pcs_label_np=inputs[1]
            watch_yaw_gt=pcs_label_np[:,7]
            for idx,pc_cube in enumerate(pcs_np):
                coordinates=np.array(np.where(pc_cube==1)[0:3]).transpose()
                print("Points in cube:{}".format(int(pc_cube.sum())))
                points=coordinates-(np.array(cfg.CUBIC_SIZE)/2)
                pcs_label_np[idx][4:7] /= np.array(cfg.CUBIC_RES)
                pcs_label_np[idx][1:4] /= (np.array(cfg.CUBIC_SIZE)/2)
                watch_a=pcs_label_np[idx][7:8]
                yaw_component=self.decompose_yaw(pcs_label_np[idx][7:8])
                yaw = np.arctan2(yaw_component[:,1], yaw_component[:,0])
                delta=pcs_label_np[idx][7:8]-yaw
                box_one=pcs_label_np[idx].copy()
                box_one[7]= yaw
                pcd_vispy_standard(scans=points, boxes=box_one, point_size=0.1)

        elif type=='output':
            net_outs=inputs

            cube_score_det = net_outs[0:self.arg.batch_size,0:2]
            cube_cls_det=(cube_score_det[:,0]<cube_score_det[:,1]).astype(np.float32).reshape(-1,1)

            cube_size_det = net_outs[0:self.arg.batch_size,2:5]+self.arg.car_size

            cube_ctr_det = net_outs[0:self.arg.batch_size,5:8]

            cube_yaw_component_det= net_outs[0:self.arg.batch_size,8:10]

            yaw = np.arctan2(cube_yaw_component_det[:,1], cube_yaw_component_det[:,0])

            cube_yaw_det = yaw.reshape(-1,1)

            return np.hstack((cube_cls_det,np.zeros_like(cube_ctr_det),cube_size_det,cube_yaw_det))

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
            # print "real angle",angle
            rotated = cv2.warpAffine(image, M,(cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1]),flags=cv2.INTER_NEAREST)
            # watch1 = image.sum(axis=-1, ).reshape(30, 30, 1)
            # watch2 = rotated.sum(axis=-1,).reshape(30,30,1)
            # a = image.sum()
            # b = rotated.sum()
            # cv2.namedWindow('a1',cv2.WINDOW_NORMAL)
            # cv2.namedWindow('a2',cv2.WINDOW_NORMAL)
            # cv2.imshow('a1', watch1)
            # cv2.imshow('a2',watch2)
            # cv2.waitKey()
            return rotated

        # print "key: ",key,"angle",angle
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

    def cube_augmentation(self, cube_array,aug_data=True, DEBUG=False):
        processor_cnt = self.arg.multi_process
        batch_size = cube_array.shape[0]
        assert batch_size % processor_cnt == 0, 'BatchSize must be multiples of {}!'.format(processor_cnt)
        delta = batch_size / processor_cnt
        mgr = multiprocessing.Manager()
        result_dict = mgr.dict()
        # result_dict = dict({})
        data_list = []
        processor_list = []
        angel_out = []
        scalar_out = []
        translation_out = []
        for i in range(processor_cnt):
            if aug_data:
                angel = (random.rand() - 0.500) * np.pi * 0.9
                # angel = 0
                scalar = 1.0+(random.rand()-0.5)*0.08
                # scalar = 0.5
                x= (random.rand(1)-0.5) * 1.4
                y= (random.rand(1)-0.5) * 1.4
                z= (random.rand(1)-0.5) * 0.7
                translation = np.array([x,y,z]).reshape(3,1)
            else:
                angel = 0.0
                scalar = 1.0
                translation = np.zeros([3, 1], dtype=np.float32)

            angel_out.append(angel)
            scalar_out.append(scalar)
            translation_out.append(translation)
            points_min = cube_array[delta * i:delta * i + delta]
            center = (cfg.CUBIC_SIZE[0] / 2, cfg.CUBIC_SIZE[1] / 2)
            # cube_train.cv2_rotation_trans(result_dict, i, points_min, center, angel, scalar, translation)
            processor = multiprocessing.Process(target=cube_train.cv2_rotation_trans,
                                                args=(result_dict, i, points_min, center, angel, scalar, translation))
            processor_list.append(processor)

        map(multiprocessing.Process.start, processor_list)
        map(multiprocessing.Process.join, processor_list)
        for i in range(processor_cnt):
            data_list.append(result_dict[i])
        stack_cube = np.array(data_list).reshape(-1, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 1)

        return stack_cube,[translation_out,scalar_out,angel_out]

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
            net_outs = self.network.outputs
            cube_score_det = self.network.outputs[:,0:2]
            cube_cls_label = self.network.cube_cls_label

            cube_size_scale_det = self.network.outputs[0:self.arg.batch_size,2:5]
            cube_size_scale_label = self.network.cube_size_label - self.arg.car_size  # normalized
            cube_ctr_det = self.network.outputs[0:self.arg.batch_size,5:8]
            cube_ctr_label = self.network.cube_ctr_label
            cube_yaw_component_det= self.network.outputs[0:self.arg.batch_size,8:10]
            cube_yaw_component_label= self.network.cube_yaw_component_label

            if self.arg.focal_loss:
                alpha = [1.0, 1.0]
                gamma = 2
                cube_probi = tf.nn.softmax(cube_score_det)
                tmp = tf.one_hot(cube_cls_label, depth=2) * ((1 - cube_probi) ** gamma) * tf.log([cfg.EPS, cfg.EPS] + cube_probi) * alpha
                cube_cross_entropy = tf.reduce_mean(-tf.reduce_sum(tmp, axis=1))
            else:
                cube_probi = tf.nn.softmax(cube_score_det)  # use for debug
                tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cube_score_det, labels=cube_cls_label)
                cube_cross_entropy = tf.reduce_mean(tmp)

            cube_size_loss=self.huber_loss(cube_size_scale_det - cube_size_scale_label,delta=1)
            cube_ctr_loss =self.huber_loss(cube_ctr_det - cube_ctr_label,delta=2)
            cube_yaw_loss =self.huber_loss(cube_yaw_component_det - cube_yaw_component_label,delta=0.5)

            loss = cube_cross_entropy*self.ratio[0]+cube_size_loss*self.ratio[1]+cube_ctr_loss*self.ratio[2]+cube_yaw_loss*self.ratio[3]
            # loss = cube_size_loss*self.ratio[1]

        with tf.name_scope('train_op'):
            global_step = tf.Variable(1, trainable=False, name='Global_Step')
            lr = tf.train.exponential_decay(self.arg.lr, global_step, 1000, 0.90, name='decay-Lr')
            train_op = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(loss, global_step=global_step)

        with tf.name_scope('train_cubic'):
            extractor_int = self.network.extractor_int
            extractor_float = self.network.extractor_weighs_float
            extractor_outs=self.network.extractor_outs  # (160, 30, 30, 15, 32)
            extractor_F_grad = tf.gradients(loss, extractor_float)
            extractor_Int_grad = tf.gradients(loss, extractor_int)
            # conv1_grad = tf.gradients(loss, self.network.conv1)
            # conv2_grad = tf.gradients(loss, self.network.conv2)
            # conv3_grad = tf.gradients(loss, self.network.conv3)
            # fc1_grad = tf.gradients(loss, self.network.fc1)
            # fc2_grad = tf.gradients(loss, self.network.fc2)
            watch_data_idx=0
            inputs_cube=tf.reshape(tf.reduce_sum(tf.squeeze(self.network.cube_input[watch_data_idx,...]),axis=-1,keep_dims=True),[-1,30,30,1])
            tf.summary.image('extractor_int', tf.reshape(extractor_int, [1, 125, -1, 1]))
            data0_kernel0_outs=tf.transpose(tf.reshape(extractor_outs[0,:,:,2,:],[1,30,30,-1]),[3,1,2,0])
            data0_kernel1_outs=tf.transpose(tf.reshape(extractor_outs[1,:,:,2,:],[1,30,30,-1]))
            data0_kernel2_outs=tf.transpose(tf.reshape(extractor_outs[2,:,:,2,:],[1,30,30,-1]))
            data0_kernel3_outs=tf.transpose(tf.reshape(extractor_outs[3,:,:,2,:],[1,30,30,-1]))

            tf.summary.image('extractor_inputs_cube', inputs_cube)
            tf.summary.image('extractor_outs1', data0_kernel0_outs,max_outputs=50)
            # tf.summary.image('extractor_outs2', data0_kernel1_outs,max_outputs=50)
            # tf.summary.image('extractor_outs3', data0_kernel2_outs,max_outputs=50)
            # tf.summary.image('extractor_outs2', data0_kernel3_outs,max_outputs=50)

            # tf.summary.image('extractor_two', tf.reshape(tf.transpose(extractor_int),[32,9,3,1]))
            # tf.summary.image('extractor_float', tf.reshape(extractor_float, [-1, 27, 32, 1]))
            # tf.summary.image('conv1_kernel', tf.reshape(self.network.conv1[0], [-1, 27, 32, 1]), max_outputs=3)
            # tf.summary.image('conv2_kernel', tf.reshape(self.network.conv2[0], [-1, 27, 64, 1]), max_outputs=3)
            # tf.summary.image('conv3_kernel', tf.reshape(self.network.conv3[0], [-1, 27, 128, 1]), max_outputs=3)
            tf.summary.histogram('extractor_int', extractor_int)
            tf.summary.histogram('extractor_float', extractor_float)

            tf.summary.histogram('float_grad', extractor_F_grad)
            tf.summary.histogram('int_grad', extractor_Int_grad)
            # tf.summary.histogram('conv1_grad', conv1_grad[0])
            # tf.summary.histogram('conv2_grad', conv2_grad[0])
            # tf.summary.histogram('conv3_grad', conv3_grad[0])
            # tf.summary.histogram('fc1_grad', fc1_grad[0])
            # tf.summary.histogram('fc2_grad', fc2_grad[0])

            tf.summary.scalar('0total_loss', loss)
            tf.summary.scalar('1cls_loss', cube_cross_entropy)
            tf.summary.scalar('2size_loss', cube_size_loss)
            tf.summary.scalar('3ctr_loss', cube_ctr_loss)
            tf.summary.scalar('4yaw_loss', cube_yaw_loss)
            glb_var = tf.global_variables()
            # for var in glb_var:
            # tf.summary.histogram(var.name, var)
            merged_op = tf.summary.merge_all()

        with tf.name_scope('valid_cubic'):
            epoch_cubic_recall = tf.placeholder(dtype=tf.float32)
            cubic_recall_smy_op = tf.summary.scalar('0cubic_recall', epoch_cubic_recall)
            epoch_cubic_precise = tf.placeholder(dtype=tf.float32)
            cubic_precise_smy_op = tf.summary.scalar('0cubic_precise', epoch_cubic_precise)

            epoch_extractorP_occupy = tf.placeholder(dtype=tf.float32)
            cubic_occupyP_smy_op = tf.summary.scalar('1extractor_occupy_positive', epoch_extractorP_occupy)
            epoch_extractorN_occupy = tf.placeholder(dtype=tf.float32)
            cubic_occupyN_smy_op = tf.summary.scalar('2extractor_occupy_negative', epoch_extractorN_occupy)

            epoch_size_evalue = tf.placeholder(dtype=tf.float32)
            epoch_size_evalue_smy_op = tf.summary.scalar('3size_evalue', epoch_size_evalue)
            epoch_yaw_evalue = tf.placeholder(dtype=tf.float32)
            epoch_yaw_evalue_smy_op = tf.summary.scalar('3yaw_evalue', epoch_yaw_evalue)
            epoch_ctr_evalue = tf.placeholder(dtype=tf.float32)
            epoch_ctr_evalue_smy_op = tf.summary.scalar('3ctr_evalue', epoch_ctr_evalue)
            valid_summary_op = tf.summary.merge([cubic_recall_smy_op, cubic_precise_smy_op, cubic_occupyP_smy_op,
                                                 cubic_occupyN_smy_op,epoch_size_evalue_smy_op,epoch_yaw_evalue_smy_op,
                                                 epoch_ctr_evalue_smy_op])

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
        cube_label_gt = np.concatenate((np.ones([self.arg.batch_size]), np.zeros([self.arg.batch_size]))).astype(np.int32)
        train_epoch_cnt = int(np.ceil(float(self.dataset.train_positive_cube_cnt) / self.arg.batch_size))
        training_series = range(train_epoch_cnt)  # range(train_epoch_cnt)  # train_epoch_cnt
        for epo_cnt in range(self.arg.epoch_iters):
            for data_idx in training_series:
                iter = global_step.eval()
                timer.tic()
                series = self.train_series_Gen(self.arg.batch_size, 'train')
                data_batchP,data_batch_label = self.dataset.get_minibatch(series[0], data_type='train', classify='positive')
                data_batchN = self.dataset.get_minibatch(series[1], data_type='train', classify='negative')
                data_batch = np.vstack((data_batchP, data_batchN))
                timer.toc()
                time1 = timer.average_time

                timer.tic()
                if self.arg.use_aug_data_method:
                    tile_cnt=self.arg.batch_size*2/self.arg.multi_process
                    data_aug,new_label = self.cube_augmentation(data_batch,aug_data=True, DEBUG=False)
                    data_batch_label_cp = data_batch_label.copy()
                    # yaw
                    yaw_aug=np.hstack(np.tile(np.array(new_label[2])[0:self.arg.multi_process/2,...],[tile_cnt,1]).transpose())
                    data_batch_label[:, 7]=data_batch_label[:,7]+yaw_aug
                    # size
                    size_aug=np.hstack(np.tile(np.array(new_label[1])[0:self.arg.multi_process/2,...],[tile_cnt,1]).transpose()).reshape(-1,1)
                    data_batch_label[:, 4:7]=data_batch_label[:,4:7] * size_aug
                    # ctr
                    ctr_aug=np.squeeze(np.array(new_label[0]))
                    ctr_aug = np.tile(ctr_aug[0:self.arg.multi_process/2,...], [tile_cnt, 1, 1])
                    # b = ctr_aug.transpose(1, 2, 0)
                    ctr_aug = np.hstack(ctr_aug.transpose(1, 2, 0)).transpose()
                    data_batch_label[:, 1:4]= ctr_aug
                else:
                    data_batch_label_cp=data_batch_label.copy()
                    data_aug = data_batch
                    data_batch_label[:, 1:4] = np.zeros_like(data_batch_label[:, 1:4])
                timer.toc()
                time2 = timer.average_time

                if DEBUG:
                    a = data_batch.sum()
                    b = data_aug.sum()
                    if a != b:
                        print 'There is some points loss in data-aug'
                    else:
                        print 'points cnt: ', a," ",b
                    # origin_yaw=data_batch_label_cp[:,7]
                    # changed_yaw=data_batch_label[:,7]
                    # delta_yaw=origin_yaw-changed_yaw
                    # print origin_yaw
                    # print changed_yaw
                    # print delta_yaw

                    outs=self.format_convertor(type='input', inputs=[data_aug[0:self.arg.batch_size,...],data_batch_label])

                feed_dict = {self.network.cube_input: data_aug,
                             self.network.cube_cls_label: cube_label_gt,
                             self.network.cube_size_label: data_batch_label[:,4:7],  # type,xyz,lwh,yaw,
                             self.network.cube_ctr_label: data_batch_label[:,1:4],
                             self.network.cube_yaw_component_label: self.decompose_yaw(data_batch_label[:,7]),
                             }
                timer.tic()

                net_outs_,extractor_outs_,extractor_int_, extractor_float_, \
                cube_probi_, cube_size_scale_det_,cube_ctr_det_,cube_yaw_component_det_, \
                loss_,cube_cross_entropy_,cube_size_loss_, cube_ctr_loss_,cube_yaw_loss_,merge_op_, _ = \
                    sess.run([net_outs,extractor_outs, extractor_int, extractor_float,
                              cube_probi,cube_size_scale_det,cube_ctr_det,cube_yaw_component_det,
                              loss,cube_cross_entropy,cube_size_loss,cube_ctr_loss,cube_yaw_loss, merged_op, train_op], feed_dict=feed_dict)
                timer.toc()
                boxes_det = self.format_convertor(type='output', inputs=net_outs_)
                self.format_convertor(type='input', inputs=[data_aug[0:self.arg.batch_size,...],boxes_det])

                if iter % 4 == 0:
                    predict_result = cube_probi_.argmax(axis=1)
                    one_train_hist = fast_hist(cube_label_gt, predict_result)
                    occupy_part_pos = (extractor_int_.reshape(-1) == 1.0).astype(float).sum() / extractor_int_.size
                    occupy_part_neg = (extractor_int_.reshape(-1) == -1.0).astype(float).sum() / extractor_int_.size
                    if iter==4:
                        extractor_int_pre_= extractor_int_
                    flip_cnt=np.abs(extractor_int_-extractor_int_pre_).sum()
                    extractor_int_pre_=extractor_int_
                    print 'TrainingRecord: epoch:{:d} step: {:d} loss:{:.4f} cls:{:.4f} size:{:.4f} ctr:{:.4f} yaw:{:.4f} flip: {:d} occupy: +{}% vs -{}%'. \
                        format(epo_cnt+1,iter, loss_,cube_cross_entropy_,cube_size_loss_,cube_ctr_loss_,cube_yaw_loss_,int(flip_cnt), int(occupy_part_pos * 100), int(occupy_part_neg * 100))
                    # print('    class bg precision = {:.3f}  recall = {:.3f}'.format(
                    #     (one_train_hist[0, 0] / (one_train_hist[0, 0] + one_train_hist[1, 0] + 1e-6)),
                    #     (one_train_hist[0, 0] / (one_train_hist[0, 0] + one_train_hist[0, 1] + 1e-6))))
                    print '    cls car precision = {:.3f}  recall = {:.3f}'.format(
                        (one_train_hist[1, 1] / (one_train_hist[1, 1] + one_train_hist[0, 1] + 1e-6)),
                        (one_train_hist[1, 1] / (one_train_hist[1, 1] + one_train_hist[1, 0] + 1e-6))), '\n'
                    if socket.gethostname() == "szstdzcp0325" and False:
                        with self.printoptions(precision=2, suppress=False, linewidth=10000):
                            print 'scores: {}'.format(cube_probi_[:, 1])
                            print 'divine:', str(predict_result)
                            print 'labels:', str(cube_label_gt), '\n'

                if iter % 2 == 0 and cfg.TRAIN.TENSORBOARD:
                    pass
                    self.writer.add_summary(merge_op_, iter)

                if (iter % 3000 == 0 and cfg.TRAIN.DEBUG_TIMELINE) or iter == 150:
                    if socket.gethostname() == "szstdzcp0325":
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _ = sess.run([cube_score_det], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                        # chrome://tracing
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        file_name_=os.path.join(self.current_saver_path,'..','training-step-' + str(iter).zfill(7) + '.ctf.json')
                        trace_file = open(file_name_, 'w')
                        trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                        trace_file.close()

            if epo_cnt % 10 == 0 and cfg.TRAIN.EPOCH_MODEL_SAVE:
                pass
                self.snapshot(sess, epo_cnt)
            if cfg.TRAIN.USE_VALID and False:
                with tf.name_scope('valid_cubic_' + str(epo_cnt + 1)):
                    print 'Valid the net at the end of epoch: {} ...'.format(epo_cnt + 1)
                    hist = np.zeros((cfg.NUM_CLASS, cfg.NUM_CLASS), dtype=np.float32)
                    total_size_loss=0
                    total_ctr_loss=0
                    total_yaw_loss=0

                    valid_epoch_cnt = int(self.dataset.valid_positive_cube_cnt / self.arg.batch_size)
                    for data_idx in range(valid_epoch_cnt):
                        series = self.train_series_Gen(self.arg.batch_size, 'valid')
                        data_batchP,data_batch_label= self.dataset.get_minibatch(series[0], data_type='valid', classify='positive')
                        data_batchN = self.dataset.get_minibatch(series[1], data_type='valid', classify='negative')
                        data_batch = np.vstack((data_batchP, data_batchN))

                        feed_dict_ = {self.network.cube_input: data_batch,
                                     self.network.cube_cls_label: cube_label_gt,
                                     self.network.cube_size_label: data_batch_label[:, 4:7],  # type,xyz,lwh,yaw,
                                     self.network.cube_ctr_label: np.zeros_like(data_batch_label[:, 1:4]),
                                     self.network.cube_yaw_component_label: self.decompose_yaw(data_batch_label[:, 7]),
                                         }
                        valid_cls_score_,loss_,cube_cross_entropy_,cube_size_loss_, cube_ctr_loss_,cube_yaw_loss_\
                            = sess.run([cube_score_det,loss,cube_cross_entropy,cube_size_loss,cube_ctr_loss,cube_yaw_loss,]
                                       , feed_dict=feed_dict_)
                        total_yaw_loss+=cube_yaw_loss_/valid_epoch_cnt
                        total_ctr_loss+=cube_ctr_loss_/valid_epoch_cnt
                        total_size_loss+=cube_size_loss_/valid_epoch_cnt

                        valid_result = valid_cls_score_.argmax(axis=1)
                        one_hist = fast_hist(cube_label_gt, valid_result)
                        hist += one_hist
                        if cfg.TRAIN.VISUAL_VALID:
                            print 'Valid step: {:d}/{:d}'.format(data_idx + 1, valid_epoch_cnt)
                            # print('    class bg precision = {:.3f}  recall = {:.3f}'.format(
                            #     (one_hist[0, 0] / (one_hist[0, 0] + one_hist[1, 0] + 1e-6)),
                            #     (one_hist[0, 0] / (one_hist[0, 0] + one_hist[0, 1] + 1e-6))))
                            print('    class car precision = {:.3f}  recall = {:.3f}'.format(
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[0, 1] + 1e-6)),
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[1, 0] + 1e-6))))
                        if data_idx % 20 == 0 and cfg.TRAIN.TENSORBOARD:
                            pass
                            # train_writer.add_summary(valid_result_, data_idx/20+epo_cnt*1000)
                valid_extractor_int_ = sess.run(extractor_int)
                occupy_part_pos = (valid_extractor_int_.reshape(-1) == 1.0).astype(float).sum() / valid_extractor_int_.size
                occupy_part_neg = (valid_extractor_int_.reshape(-1) == -1.0).astype(float).sum() / valid_extractor_int_.size

                precise_total = hist[1, 1] / (hist[1, 1] + hist[0, 1] + 1e-6)
                recall_total = hist[1, 1] / (hist[1, 1] + hist[1, 0] + 1e-6)
                valid_res = sess.run(valid_summary_op, feed_dict={epoch_cubic_recall: recall_total,
                                                                  epoch_cubic_precise: precise_total,
                                                                  epoch_extractorP_occupy: occupy_part_pos,
                                                                  epoch_extractorN_occupy: occupy_part_neg,
                                                                  epoch_size_evalue: occupy_part_neg,
                                                                  epoch_yaw_evalue: occupy_part_neg,
                                                                  epoch_ctr_evalue: occupy_part_neg,
                                                                  })
                self.writer.add_summary(valid_res, epo_cnt + 1)
                print 'Validation of epoch_{}: cubic_precision = {:.3f}  cubic_recall = {:.3f} size_evalue: {:.4f} ctr_evalue: {:.4f} yaw_evalue: {:.4f}\n' \
                    .format(epo_cnt + 1, precise_total, recall_total,total_size_loss,total_ctr_loss,total_yaw_loss)
            self.shuffle_series()
        print yellow('Training process has been done! Claude,you need to have a break!')

class cube_test(object):
    def __init__(self, arg_,dataset_,network_):
        self.arg = arg_
        self.dataset = dataset_
        self.network =network_

        self.saver = tf.train.Saver(max_to_keep=1000)

    def decompose_yaw(self,yaw_array):
        #//see the notebook page 1

        #make the theta in [-pi,_pi]
        indice1=np.where(yaw_array>np.pi)
        yaw_array[indice1] -= 2*np.pi
        indice2=np.where(yaw_array< -np.pi)
        yaw_array[indice2]+= 2*np.pi

        #make the theta in [-pi/2,_pi/2] #TODO:declaration:assuming that the car oriente forward
        indice1=np.where(yaw_array>np.pi/2)
        yaw_array[indice1] -= np.pi
        indice2=np.where(yaw_array< -np.pi/2)
        yaw_array[indice2]+= np.pi

        x_ary=np.cos(yaw_array).reshape(-1,1)
        y_ary=np.sin(yaw_array).reshape(-1,1)
        ret = np.hstack((x_ary, y_ary))
        # yaw = np.arctan2(y_ary,x_ary)
        # #TODO:hxd:todo: reduce the half angle
        # differ=yaw_array.reshape(-1)-yaw.reshape(-1)

        return ret

    def format_convertor(self,type,inputs):
        if type=='input':
            pcs_np=inputs[0]
            pcs_label_np=inputs[1]
            watch_yaw_gt=pcs_label_np[:,7]
            for idx,pc_cube in enumerate(pcs_np):
                coordinates=np.array(np.where(pc_cube==1)[0:3]).transpose()
                print("Points in cube:{}".format(int(pc_cube.sum())))
                points=coordinates-(np.array(cfg.CUBIC_SIZE)/2)
                pcs_label_np[idx][4:7] /= np.array(cfg.CUBIC_RES)
                pcs_label_np[idx][1:4] /= (np.array(cfg.CUBIC_SIZE)/2)
                watch_a=pcs_label_np[idx][7:8]
                yaw_component=self.decompose_yaw(pcs_label_np[idx][7:8])
                yaw = np.arctan2(yaw_component[:,1], yaw_component[:,0])
                delta=pcs_label_np[idx][7:8]-yaw
                box_one=pcs_label_np[idx].copy()
                box_one[7]= yaw
                pcd_vispy_standard(scans=points, boxes=box_one, point_size=0.1)

        elif type=='output':
            det_outs=inputs[0]
            labels=inputs[1]
            labels_ctr=labels[:,1:4]

            cube_score_det = det_outs[:,0:2]
            cube_cls_det=(cube_score_det[:,0]<cube_score_det[:,1]).astype(np.float32).reshape(-1,1)

            cube_size_det = det_outs[:,2:5]+self.arg.car_size

            # cube_ctr_det = det_outs[:,5:8]+labels_ctr
            cube_ctr_det = labels_ctr

            cube_yaw_component_det= det_outs[:,8:10]

            yaw = np.arctan2(cube_yaw_component_det[:,1], cube_yaw_component_det[:,0])

            cube_yaw_det = yaw.reshape(-1,1)

            return np.hstack((cube_cls_det,cube_ctr_det,cube_size_det,cube_yaw_det))

        elif type=='valid':
            pc=inputs[0]
            boxes_det=inputs[1]
            pcd_vispy_standard(scans=pc, boxes=boxes_det, point_size=0.05)
            pass

    def label2hangzhou_label(self,label):
        ry=-1*(label["ry"].reshape(-1,1)+np.pi/2)
        boxes=label["boxes_3D"].reshape(-1,6)#xyz lwh

        box2d=label["boxes"].reshape(-1,4)
        type=np.ones_like(ry)
        new_lbael=np.hstack((type,boxes,ry,box2d))

        return new_lbael

    def cubic_rpn_grid_pyfc(self,lidarPoints, rpnBoxes):
        # rpnBoxes:(x1,y1,z1),(x2,y2,z2),cls_label,yaw

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

            return filter_points, np.array([x_min, y_min, z_min, 0.], dtype=np.float32), np.array(
                [box[0], box[1], box[2], 0.], dtype=np.float32)

        cubic_size = [cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 1]
        res = []
        display_stack = []

        if DEBUG:
            pass
            display_stack.append(pcd_vispy_standard(lidarPoints, boxes=rpnBoxes, visible=False, multi_vis=True))

        for iidx, box in enumerate(rpnBoxes):
            rpn_points, min_vertex, ctr_vertex = bounding_filter(lidarPoints, box[1:])
            points_mv_min = np.subtract(rpn_points, min_vertex)  # using fot coordinate
            points_mv_ctr = np.subtract(rpn_points, ctr_vertex)  # using as feature

            x_cub = np.divide(points_mv_min[:, 0], cfg.CUBIC_RES[0]).astype(np.int32)
            y_cub = np.divide(points_mv_min[:, 1], cfg.CUBIC_RES[1]).astype(np.int32)
            z_cub = np.divide(points_mv_min[:, 2], cfg.CUBIC_RES[2]).astype(np.int32)
            # feature = np.hstack((np.ones([len(points_mv_ctr[:,3]),1]),points_mv_ctr[:,3:]))
            feature = np.ones([len(points_mv_ctr[:, 3]), 1], dtype=np.float32)

            cubic_feature = np.zeros(shape=cubic_size, dtype=np.float32)
            cubic_feature[x_cub, y_cub, z_cub] = feature  # TODO:select&add feature # points_mv_ctr  # using center coordinate system
            res.append(cubic_feature)

            if DEBUG:
                box_mv = [box[0] - box[0], box[1] - box[1], box[2] - box[2], cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1],
                          cfg.CUBIC_SIZE[2], 1, 0, 0]
                box_gt_mv = [box[0] - box[0], box[1] - box[1], box[2] - box[2], cfg.ANCHOR[0], cfg.ANCHOR[1],
                             cfg.ANCHOR[2], 1, 0, 0]

                cubic_feature = np.hstack((x_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[0]/2),y_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[1]/2),z_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[2]/2))) #points_mv_ctr_rot

                display_stack.append(pcd_vispy_standard(cubic_feature.reshape(-1, 3), name='grid_' + str(iidx),
                                               boxes=box_mv, visible=False, point_size=0.1,
                                               multi_vis=True))
                display_stack.append(pcd_vispy_standard(points_mv_ctr.reshape(-1, 4), name='origin_' + str(iidx),
                                               boxes=box_gt_mv, visible=False, point_size=0.1,
                                               multi_vis=True))
            # break
        if DEBUG:
            pcd_show_now()
        stack_size = np.concatenate((np.array([-1]), cubic_size))
        return np.array(res, dtype=np.float32).reshape(stack_size)

    def testing(self, sess):
        net_outs = self.network.outputs
        with tf.name_scope('load_weights'):
            sess.run(tf.global_variables_initializer())
            if self.arg.weights is not None:
                self.network.load_weigths(self.arg.weights, sess, self.saver)
                print 'Loading trained model weights from {:s}'.format(red(self.arg.weights))
            else:
                print 'NO weights found!'
                exit(0xf1)

        timer = Timer()
        if DEBUG:
            pass
            vispy_init()

        for data_idx in range(200):
            timer.tic()
            data_label, lidar_np= self.dataset.get_minibatch(data_idx)

            if data_label["gt_classes"].size==0:
                continue
            data_label_new= self.label2hangzhou_label(data_label)
            # pcd_vispy_standard(scans=lidar_np,boxes=data_label_new)
            cube_data = self.cubic_rpn_grid_pyfc(lidar_np,data_label_new)

            # outs=self.format_convertor(type='input', inputs=[cube_data,data_label_new])

            feed_dict = {self.network.cube_input: cube_data,
                         }
            timer.tic()
            net_outs_= sess.run(net_outs, feed_dict=feed_dict)
            timer.toc()
            boxes_det = self.format_convertor(type='output', inputs=[net_outs_,data_label_new])
            self.format_convertor(type='valid', inputs=[lidar_np,boxes_det])

        print yellow('Training process has been done! Claude,you need to have a break!')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a CombineNet network')
    parser.add_argument('--lr', dest='lr', default=0.02, type=float)
    parser.add_argument('--epoch_iters', dest='epoch_iters', default=1000, type=int)
    parser.add_argument('--imdb_type', dest='imdb_type', choices=['kitti', 'hangzhou'], default='kitti', type=str)
    parser.add_argument('--use_demo', action='store_true')
    parser.add_argument('--focal_loss', action='store_true')
    parser.add_argument('--use_aug_data_method', action='store_true')
    parser.add_argument('--one_piece', action='store_true')
    parser.add_argument('--task_name', action='store_true')
    parser.add_argument('--weights', dest='weights', default=None, type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # args_input = parse_args()
    arg = edict()
    arg.lr = 0.001
    arg.amplifier_rate= 100
    arg.epoch_iters = 1000
    arg.car_size = np.array([3.88311640418,1.62856739989,1.52563191462])
    arg.loss_weights= [2,1,0,1]# cls,size,ctr,yaw
    arg.imdb_type = 'kitti'
    arg.use_demo = True
    arg.weights = "/home/likewise-open/SENSETIME/hexindong/ProjectDL/cubic-local/process_record/test_K-5_D0_Data_aug/CubeOnly_epoch_110.ckpt"
    arg.focal_loss = True
    arg.use_aug_data_method = False
    arg.positive_points_needed = 40
    arg.one_piece = True
    arg.task_name = 'cube_3state_K-5_D0_data_aug'
    train=False
    if socket.gethostname() == "szstdzcp0325":
        arg.batch_size = 20 # will be double when adding negative batch-data
        arg.multi_process = 4
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        arg.batch_size = 150  # will be double when adding negative batch-data
        arg.multi_process = 20

    if socket.gethostname() == "szstdzcp0325":
        cube_data_path = '/home/likewise-open/SENSETIME/hexindong/DISK1/DATASET/KITTI/object/CUBE/gt_filtered_cubes'
        # cube_data_path = '/home/likewise-open/SENSETIME/hexindong/DISK1/DATASET/KITTI/object/CUBE/small_gt_filtered_cubes'
    else:
        cube_data_path = '/mnt/lustre/hexindong/DataSet/KITTI/object/CUBE/gt_filtered_cubes'
        # cube_data_path = '/mnt/lustre/hexindong/DataSet/KITTI/object/CUBE/small_gt_filtered_cubes'

    if train:
        DataSet = cube_filter_data_load(cube_data_path, arg, one_piece=arg.one_piece)
    else:
        path_="/home/likewise-open/SENSETIME/hexindong/DISK1/DATASET/KITTI/object/training"
        DataSet= DataLoader(path_,("__background__","Car"))

    if train:
        NetWork = net_build([64, 64, 128, 128, 64, 10], arg)
    else:
        NetWork = net_build([64, 64, 128, 128, 64, 10], arg)

    if train:
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            task = cube_train(arg,DataSet,NetWork)
            task.training(sess)
    else:
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            task = cube_test(arg, DataSet,NetWork)
            task.testing(sess)
