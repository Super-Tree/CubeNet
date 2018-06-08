# coding=utf-8
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
from tools.printer import red,blue,yellow,cyan
from easydict import EasyDict as edict
from boxes_factory import box_np_view

import multiprocessing
import cv2
DEBUG = True

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


class data_load(object):
    """
    valid negative  46260  positive  5940
    train negative 239576  positive 30924
    """

    def __init__(self, path, one_piece=False):
        self.path = path
        self.train_positive_cube_cnt = 30924
        self.train_negative_cube_cnt = 239576
        self.valid_positive_cube_cnt = 5940
        self.valid_negative_cube_cnt = 46260

        self.load_all_data = False
        self.TrainSet_POS = []
        self.TrainSet_NEG = []
        self.ValidSet_POS = []
        self.ValidSet_NEG = []
        if one_piece:
            self.eat_data_in_one_piece()
            self.load_all_data = True

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
        TrainSet_POS_file_name = path_add(self.path, 'data_in_one_piece', 'TrainSet_POS.npy')
        TrainSet_NEG_file_name = path_add(self.path, 'data_in_one_piece', 'TrainSet_NEG.npy')
        ValidSet_POS_file_name = path_add(self.path, 'data_in_one_piece', 'ValidSet_POS.npy')
        ValidSet_NEG_file_name = path_add(self.path, 'data_in_one_piece', 'ValidSet_NEG.npy')
        if os.path.exists(TrainSet_POS_file_name) and os.path.exists(ValidSet_NEG_file_name) and os.path.exists(TrainSet_NEG_file_name)and os.path.exists(ValidSet_POS_file_name):
            self.TrainSet_POS = np.load(TrainSet_POS_file_name)
            self.TrainSet_NEG = np.load(TrainSet_NEG_file_name)
            self.ValidSet_POS = np.load(ValidSet_POS_file_name)
            self.ValidSet_NEG = np.load(ValidSet_NEG_file_name)
            print(cyan('Load data from zip file in folder:data_in_one_piece.'))
            return None

        train_pos_name_list = sorted(os.listdir(path_add(self.path, 'KITTI_TRAIN_BOX','POSITIVE')))
        train_neg_name_list = sorted(os.listdir(path_add(self.path, 'KITTI_TRAIN_BOX','NEGATIVE')))
        valid_pos_name_list = sorted(os.listdir(path_add(self.path, 'KITTI_VALID_BOX','POSITIVE')))
        valid_neg_name_list = sorted(os.listdir(path_add(self.path, 'KITTI_VALID_BOX','NEGATIVE')))
        print(yellow('Let`s eating !'))
        for name in train_pos_name_list:
            self.TrainSet_POS.append(np.load(path_add(self.path, 'KITTI_TRAIN_BOX','POSITIVE')+'/'+name))
        self.TrainSet_POS = np.array(self.TrainSet_POS, dtype=np.uint8)
        np.save(TrainSet_POS_file_name,self.TrainSet_POS)
        print(blue('  Yummy!'))

        for name in train_neg_name_list:
            self.TrainSet_NEG.append(np.load(path_add(self.path, 'KITTI_TRAIN_BOX','NEGATIVE')+'/'+name))
        self.TrainSet_NEG = np.array(self.TrainSet_NEG, dtype=np.uint8)
        np.save(TrainSet_NEG_file_name, self.TrainSet_NEG)
        print(blue('  Take another piece!'))

        for name in valid_pos_name_list:
            self.ValidSet_POS.append(np.load(path_add(self.path, 'KITTI_VALID_BOX','POSITIVE')+'/'+name))
        self.ValidSet_POS = np.array(self.ValidSet_POS, dtype=np.uint8)
        np.save(ValidSet_POS_file_name, self.ValidSet_POS)
        print(blue('  One more!'))

        for name in valid_neg_name_list:
            self.ValidSet_NEG.append(np.load(path_add(self.path, 'KITTI_VALID_BOX','NEGATIVE')+'/'+name))
        self.ValidSet_NEG = np.array(self.ValidSet_NEG, dtype=np.uint8)
        np.save(ValidSet_NEG_file_name, self.ValidSet_NEG)
        print(blue('  Full ...!'))
        print(yellow('Data has been successfully loaded and writed in zip npy file!'))

class net_build(object):
    def __init__(self, channel, training=True):
        self.cube_input = tf.placeholder(dtype=tf.float32,
                                         shape=[None, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], 1])
        self.cube_label = tf.placeholder(dtype=tf.int32, shape=[None])

        with tf.variable_scope('conv3d_1', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_1 = tf.layers.Conv3D(filters=channel[0], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[1, 1, 1], padding="valid", _reuse=tf.AUTO_REUSE,
                                             _scope=scope, trainable=training)
            self.maxpool_1 = tf.layers.MaxPooling3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')
            self.bn_1 = tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

        with tf.variable_scope('conv3d_2', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_2 = tf.layers.Conv3D(filters=channel[1], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[1, 1, 1], padding="valid", _reuse=tf.AUTO_REUSE,
                                             _scope=scope, trainable=training)
            self.maxpool_2 = tf.layers.MaxPooling3D(pool_size=[2, 2, 2], strides=[2, 2, 2], padding='same')
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

        self.cube_score = self.apply(self.cube_input)

    def apply(self, inputs):
        assert len(inputs.shape.as_list()) == 5, ' The data`s dimension of the network isnot 5!'
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

        return res

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
    def __init__(self, arg, dataset, network, writer):
        self.arg = arg
        self.dataset = dataset
        self.network = network
        self.saver = tf.train.Saver(max_to_keep=100)
        self.writer = writer
        self.random_folder = cfg.RANDOM_STR

    def snapshot(self, sess, epoch_cnt=0):
        import os
        output_dir = path_add(cfg.ROOT_DIR, 'output', self.random_folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = os.path.join(output_dir, 'CubeOnly_epoch_{:d}'.format(epoch_cnt) + '.ckpt')
        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

    @staticmethod
    def cv2_rotation_trans(dict_share, key, data, center, angle, scale, translation):
        def cv2_op(image):
            image = image.reshape(cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1],-1)
            M = cv2.getRotationMatrix2D(center, angle*57.296, scale)
            rotated = cv2.warpAffine(image, M, (cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1]))
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

    def cube_augmentation(self, cube_array, aug_data=True,DEBUG=False):
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
                translation = np.zeros([3,1], dtype=np.float32)
            angel_out.append(angel)
            points_min = cube_array[delta * i:delta * i + delta]
            center = (cfg.CUBIC_SIZE[0]/2, cfg.CUBIC_SIZE[1]/2)
            # processor = multiprocessing.Process(target=cube_train.single_process_task,
            #                                     args=(result_dict, i, points_min, center, angel, scalar, translation))
            processor = multiprocessing.Process(target=cube_train.cv2_rotation_trans,
                                                args=(result_dict, i, points_min, center, angel, scalar, translation))
            processor_list.append(processor)

        map(multiprocessing.Process.start, processor_list)
        map(multiprocessing.Process.join, processor_list)
        for i in range(processor_cnt):
            data_list.append(result_dict[i])
        stack_cube = np.array(data_list).reshape(-1,cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1],cfg.CUBIC_SIZE[2],1)

        return stack_cube

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
                tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cube_score, labels=cube_label)
                cube_cross_entropy = tf.reduce_mean(tmp)

            loss = cube_cross_entropy

        with tf.name_scope('train_op'):
            global_step = tf.Variable(1, trainable=False, name='Global_Step')
            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, 10000, 0.90, name='decay-Lr')
            train_op = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(loss, global_step=global_step)

        with tf.name_scope('train_cubic'):
            tf.summary.scalar('total_loss', loss)
            glb_var = tf.global_variables()
            for var in glb_var:
                tf.summary.histogram(var.name, var)
            merged_op = tf.summary.merge_all()

        with tf.name_scope('valid_cubic'):
            epoch_cubic_recall = tf.placeholder(dtype=tf.float32)
            cubic_recall_smy_op = tf.summary.scalar('cubic_recall', epoch_cubic_recall)
            epoch_cubic_precise = tf.placeholder(dtype=tf.float32)
            cubic_precise_smy_op = tf.summary.scalar('cubic_precise', epoch_cubic_precise)

            valid_summary_op = tf.summary.merge([cubic_recall_smy_op, cubic_precise_smy_op])

        with tf.name_scope('load_weights'):
            sess.run(tf.global_variables_initializer())
            if self.arg.weights is not None:
                self.network.load_weigths(self.arg.weights, sess, self.saver)
                print 'Loading pre-trained model weights from {:s}'.format(self.arg.weights)

        timer = Timer()
        if DEBUG:
            pass
            vispy_init()
        training_series = range(10,100)  # self.epoch
        for epo_cnt in range(self.arg.epoch_iters):
            for data_idx in training_series:
                iter = global_step.eval()
                timer.tic()
                series = range(data_idx*self.arg.batch_size, (data_idx+1)*self.arg.batch_size)
                data_batch = self.dataset.get_minibatch(series, data_type='train', classify='positive')
                a = data_batch.sum()
                data_aug = self.cube_augmentation(data_batch, DEBUG=False)
                b = data_batch.sum()
                timer.toc()
                print 'Time cost of loading and processing cube minibatch: ', timer.average_time
                if DEBUG:
                    pass
                    box_np_view(data_batch[data_idx],data_aug[data_idx])
                feed_dict = {self.network.cube_input: data_aug,
                             self.network.cube_label: labels,
                             }
                timer.tic()
                cube_score_, cube_label_, loss_, merge_op_, _ = \
                    sess.run([cube_score, cube_label, loss, merged_op, train_op], feed_dict=feed_dict)
                timer.toc()

                if iter % cfg.TRAIN.ITER_DISPLAY == 0:
                    print 'Training step: {:3d} loss: {.4f} time_cost: {:3f} '.format(iter, loss_, timer.average_time)
                    print 'scores: ', str(cube_score_).translate(None, '\n')
                    print 'divine: ', str(cube_score_).translate(None, '\n')
                    print 'labels: ', str(cube_label_).translate(None, '\n'), '\n'

                if iter % 40 == 0 and cfg.TRAIN.TENSORBOARD:
                    pass
                    self.writer.add_summary(merge_op_, iter)
                if (iter % 4000 == 0 and cfg.TRAIN.DEBUG_TIMELINE) or iter == 200:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _ = sess.run([cube_score], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    # chrome://tracing
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(cfg.LOG_DIR + '/' + 'training-step-' + str(iter).zfill(7) + '.ctf.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                    trace_file.close()

            if cfg.TRAIN.EPOCH_MODEL_SAVE:
                pass
                self.snapshot(sess, epo_cnt)

            if cfg.TRAIN.USE_VALID and False:
                with tf.name_scope('valid_cubic_' + str(epo_cnt + 1)):
                    print 'Valid the net at the end of epoch_{} ...'.format(epo_cnt + 1)
                    pred_tp_cnt, gt_cnt = 0., 0.
                    hist = np.zeros((cfg.NUM_CLASS, cfg.NUM_CLASS), dtype=np.float32)

                    for data_idx in range(self.val_epoch):  # self.val_epoch
                        data_batch = self.dataset.get_minibatch(self, [1, 12, 2], 'valid', classify='positive')

                        feed_dict = {self.network.cube_input: data_aug,
                                     self.network.cube_label: labels,
                                     }

                        cubic_cls_score_, cubic_cls_labels_, recalls_ = sess.run(
                            [cubic_cls_score, cubic_cls_labels, recalls], feed_dict=feed_dict_)
                        # train_writer.add_summary(valid, data_idx)

                        pred_tp_cnt = pred_tp_cnt + recalls_[1]
                        gt_cnt = gt_cnt + recalls_[2]
                        cubic_class = cubic_cls_score_.argmax(axis=1)
                        one_hist = fast_hist(cubic_cls_labels_, cubic_class)
                        if not math.isnan(one_hist[1, 1] / (one_hist[1, 1] + one_hist[0, 1])):
                            if not math.isnan(one_hist[1, 1] / (one_hist[1, 1] + one_hist[1, 0])):
                                hist += one_hist
                        if cfg.TRAIN.VISUAL_VALID:
                            print 'Valid step: {:d}/{:d} , rpn recall = {:.3f}' \
                                .format(data_idx + 1, self.val_epoch, float(recalls_[1]) / recalls_[2])
                            print('    class bg precision = {:.3f}  recall = {:.3f}'.format(
                                (one_hist[0, 0] / (one_hist[0, 0] + one_hist[1, 0] + 1e-6)),
                                (one_hist[0, 0] / (one_hist[0, 0] + one_hist[0, 1] + 1e-6))))
                            print('    class car precision = {:.3f}  recall = {:.3f}'.format(
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[0, 1] + 1e-6)),
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[1, 0] + 1e-6))))
                        if data_idx % 20 == 0 and cfg.TRAIN.TENSORBOARD:
                            pass
                            # train_writer.add_summary(valid_result_, data_idx/20+epo_cnt*1000)

                precise_total = hist[1, 1] / (hist[1, 1] + hist[0, 1] + 1e-6)
                recall_total = hist[1, 1] / (hist[1, 1] + hist[1, 0] + 1e-6)
                recall_rpn = pred_tp_cnt / gt_cnt
                valid_res = sess.run(valid_summary, feed_dict={epoch_rpn_recall: recall_rpn,
                                                               epoch_cubic_recall: recall_total,
                                                               epoch_cubic_precise: precise_total})
                train_writer.add_summary(valid_res, epo_cnt + 1)
                print 'Validation of epoch_{}: rpn_recall {:.3f} cubic_precision = {:.3f}  cubic_recall = {:.3f}' \
                    .format(epo_cnt + 1, recall_rpn, precise_total, recall_total)
            random.shuffle(training_series)
        print yellow('Training process has done, enjoy every day !')




if __name__ == '__main__':
    DataSet = data_load('/home/hexindong/DATASET/DATA_BOXES',one_piece=True)

    NetWork = net_build([64, 128, 128, 64, 2], )

    arg = edict()
    arg.imdb_type = 'kitti'
    arg.use_demo = True
    arg.weights = None
    arg.focal_loss = True
    arg.epoch_iters = 10
    arg.batch_size = 2000
    arg.multi_process = 4
    arg.use_aug_data_method = True
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(cfg.LOG_DIR, sess.graph, max_queue=1000, flush_secs=1)
        task = cube_train(arg, DataSet, NetWork, writer)
        task.training(sess)

    batch = DataSet.get_minibatch(range(10), data_type='trian', classify='positive')
