# coding=utf-8
from tools.data_visualize import show_rpn_tf

import random
import os
import math
import numpy as np
import tensorflow as tf
from tools.timer import Timer
from network.config import cfg
from tools.utils import fast_hist
from tensorflow.python.client import timeline
from tensorflow.python import pywrap_tensorflow
from tools.data_visualize import pcd_vispy,vispy_init,pcd_vispy_client
##================================================
# from multiprocessing import Process,Queue
# MSG_QUEUE = Queue(200)
##================================================
DEBUG = False

class CubicNet_Train_sti(object):
    def __init__(self, network, data_set, args):
        self.saver = tf.train.Saver(max_to_keep=100)
        self.net = network
        self.dataset = data_set
        self.args = args
        self.random_folder = cfg.RANDOM_STR
        self.epoch = self.dataset.training_rois_length
        self.val_epoch = self.dataset.validing_rois_length

    def snapshot(self, sess, iter=None, final=False):
        output_dir = os.path.join(cfg.ROOT_DIR, 'output', self.random_folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not final:
            filename = os.path.join(output_dir, 'CubicNet_STi_iter_{:d}'.format(iter) + '.ckpt')
            self.saver.save(sess, filename)
            print 'Wrote snapshot to: {:s}'.format(filename)
        else:
            filename = os.path.join(output_dir, 'CombiNet_STi_iter_{:d}_final'.format(iter) + '.ckpt')
            self.saver.save(sess, filename)

    @staticmethod
    def modified_smooth_l1(sigma, bbox_pred, bbox_targets):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        diffs = tf.subtract(bbox_pred, bbox_targets)

        smooth_l1_sign = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(diffs, diffs), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(diffs), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
        outside_mul = smooth_l1_result

        return outside_mul

    def training(self, sess, train_writer):
        with tf.name_scope('loss_cubic'):
            cubic_cls_score = tf.reshape(self.net.get_output('cubic_cnn'), [-1, 2])
            cubic_cls_labels = tf.reshape(tf.cast(self.net.get_output('rpn_rois')[:,-2], tf.int64), [-1])

            if not cfg.TRAIN.FOCAL_LOSS:
                cubic_cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cubic_cls_score, labels=cubic_cls_labels))
            else:
                # alpha = [0.75,0.25]  # 0.25 for label=1
                gamma = 2
                cubic_cls_probability = tf.nn.softmax(cubic_cls_score)
                # formula :  Focal Loss for Dense Object Detection: FL(p)= -((1-p)**gama)*log(p)
                cubic_cross_entropy = tf.reduce_mean(-tf.reduce_sum(
                    tf.one_hot(cubic_cls_labels, depth=2) * ((1 - cubic_cls_probability) ** gamma) * tf.log(
                        [cfg.EPS, cfg.EPS] + cubic_cls_probability), axis=1))
            loss = cubic_cross_entropy

        with tf.name_scope('train_op'):
            global_step = tf.Variable(1, trainable=False, name='Global_Step')
            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, 10000, 0.996, name='decay-Lr')
            train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

        with tf.name_scope('train_cubic'):
            tf.summary.scalar('total_loss', loss)
            # bv_anchors = self.net.get_output('rpn_anchors_label')[2]
            # roi_bv = self.net.get_output('rpn_rois')[0]
            # data_bv = self.net.lidar_bv_data
            # data_gt = self.net.gt_boxes_bv
            # image_rpn = tf.reshape(show_rpn_tf(data_bv, data_gt, bv_anchors, roi_bv), (1, 601, 601, -1))
            # tf.summary.image('lidar_bv_test', image_rpn)
            glb_var = tf.global_variables()
            for i in range(len(glb_var)):
                # print glb_var[i].name
                if 'moving' not in str(glb_var[i].name):
                    if 'Adam' not in str(glb_var[i].name):
                        if 'weights' not in str(glb_var[i].name):
                            if 'rpn' not in str(glb_var[i].name):
                                if 'biases' not in str(glb_var[i].name):
                                    if 'beta'not in str(glb_var[i].name):
                                        if 'gamma' not in str(glb_var[i].name):
                                            if 'batch' not in str(glb_var[i].name):
                                                    tf.summary.histogram(glb_var[i].name, glb_var[i])
            merged = tf.summary.merge_all()

        with tf.name_scope('valid_cubic'):
            epoch_rpn_recall = tf.placeholder(dtype=tf.float32)
            rpn_recall_smy_op = tf.summary.scalar('rpn_recall', epoch_rpn_recall)
            epoch_cubic_recall = tf.placeholder(dtype=tf.float32)
            cubic_recall_smy_op = tf.summary.scalar('cubic_recall', epoch_cubic_recall)
            epoch_cubic_precise = tf.placeholder(dtype=tf.float32)
            cubic_prec_smy_op = tf.summary.scalar('cubic_precise', epoch_cubic_precise)

        sess.run(tf.global_variables_initializer())
        if self.args.fine_tune:
            if True:
                # #full graph restore
                print 'Loading pre-trained model weights from {:s}'.format(self.args.weights)
                self.net.load(self.args.weights, sess, self.saver, True)
            else:  # #part graph restore
                #  # METHOD one
                # ref_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=['vgg_feat_fc'])
                # saver1 = tf.train.Saver(ref_vars)
                # saver1.restore(sess, self.args.weights)
                #  # METHOD two
                reader = pywrap_tensorflow.NewCheckpointReader(self.args.weights)
                var_to_shape_map = reader.get_variable_to_shape_map()
                with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:
                    for key in var_to_shape_map:
                        try:
                            var = tf.get_variable(key, trainable=False)
                            sess.run(var.assign(reader.get_tensor(key)))
                            print "    Assign pretrain model: " + key
                        except ValueError:
                            print "    Ignore variable:" + key
        trainable_var_for_chk=tf.trainable_variables()#tf.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
        print 'Variables to training: ',trainable_var_for_chk

        timer = Timer()
        rpn_rois = self.net.get_output('rpn_rois')
        cubic_grid = self.net.get_output('cubic_grid')
        cubic_cnn= self.net.get_output('cubic_cnn')
        if DEBUG:
            vispy_init()  # TODO: Essential step(before sess.run) for using vispy beacuse of the bug of opengl or tensorflow
            # vision_qt = Process(target=pcd_vispy_client, args=(MSG_QUEUE,))
            # vision_qt.start()
            # print 'Process vision_qt started ...'

        training_series = range(self.epoch)  # self.epoch
        for epo_cnt in range(self.args.epoch_iters):
            for data_idx in training_series:  # DO NOT EDIT the "training_series",for the latter shuffle
                iter = global_step.eval()  # function "minimize()"will increase global_step
                blobs = self.dataset.get_minibatch(data_idx, 'train')  # get one batch
                feed_dict = {self.net.lidar3d_data: blobs['lidar3d_data'],
                             self.net.gt_boxes_3d: blobs['gt_boxes_3d']
                             }
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                timer.tic()
                cubic_cls_score_,cubic_cls_labels_,rpn_rois_,cubic_cnn_,cubic_grid_, loss_, merged_, _ = sess.run(
                    [cubic_cls_score,cubic_cls_labels,rpn_rois,cubic_cnn,cubic_grid,loss, merged, train_op],
                    feed_dict=feed_dict,options=run_options, run_metadata=run_metadata)
                timer.toc()

                cubic_result = cubic_cls_score_.argmax(axis=1)
                one_hist = fast_hist(cubic_cls_labels_, cubic_result)
                cubic_car_cls_prec = one_hist[1, 1] / (one_hist[1, 1] + one_hist[0, 1]+1e-5)
                cubic_car_cls_recall = one_hist[1, 1] / (one_hist[1, 1] + one_hist[1, 0]+1e-5)

                if iter % 1000==0 and cfg.TRAIN.DEBUG_TIMELINE:
                    #chrome://tracing
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(cfg.LOG_DIR+'/' +'training-StiData-step-'+ str(iter).zfill(7) + '.ctf.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                    trace_file.close()
                if iter % cfg.TRAIN.ITER_DISPLAY == 0:
                    print 'Iter: %d / %d, loss: %.3f' % (iter, self.args.epoch_iters * self.epoch,loss_,)
                    print 'Cubic classify precise: {:.3f}  recall: {:.3f}'.format(cubic_car_cls_prec, cubic_car_cls_recall)
                    print 'Speed: {:.3f}s / iter'.format(timer.average_time)
                    print 'divine: ', cubic_result
                    print 'labels: ', cubic_cls_labels_
                if iter % 10 == 0 and cfg.TRAIN.TENSORBOARD:
                    train_writer.add_summary(merged_, iter)
                    pass
                if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                    self.snapshot(sess, iter)
                    pass
                if DEBUG:
                    scan = blobs['lidar3d_data']
                    gt_box3d = blobs['gt_boxes_3d'][:, (0, 1, 2, 3, 4, 5, 6)]
                    gt_box3d = np.hstack((gt_box3d,np.ones([gt_box3d.shape[0],2])*4))
                    pred_boxes = np.hstack((rpn_rois_,cubic_result.reshape(-1,1)*2))
                    bbox = np.vstack((pred_boxes, gt_box3d))
                    # msg = msg_qt(scans=scan, boxes=bbox,name='CubicNet training')
                    # MSG_QUEUE.put(msg)
                    pcd_vispy(scan, boxes=bbox,name='CubicNet training')
            random.shuffle(training_series)  # shuffle the training series
            if cfg.TRAIN.USE_VALID:
                with tf.name_scope('valid_cubic_' + str(epo_cnt + 1)):
                    print 'Valid the net at the end of epoch_{} ...'.format(epo_cnt + 1)
                    # roi_bv = self.net.get_output('rpn_rois')[0]
                    # bv_anchors = self.net.get_output('rpn_anchors_label')[2]
                    # pred_rpn_ = show_rpn_tf(self.net.lidar_bv_data, self.net.gt_boxes_bv, bv_anchors, roi_bv)
                    # pred_rpn = tf.reshape(pred_rpn_,(1, 601, 601, -1))
                    # predicted_bbox = tf.summary.image('predict_bbox_bv', pred_rpn)
                    # valid_result = tf.summary.merge([predicted_bbox])
                    recalls = self.net.get_output('rpn_rois')[2]
                    pred_tp_cnt, gt_cnt = 0., 0.
                    hist = np.zeros((cfg.NUM_CLASS, cfg.NUM_CLASS), dtype=np.float32)

                    for data_idx in range(self.val_epoch):  # self.val_epoch
                        blobs = self.dataset.get_minibatch(data_idx, 'valid')
                        feed_dict_ = {
                            self.net.lidar3d_data: blobs['lidar3d_data'],
                            self.net.lidar_bv_data: blobs['lidar_bv_data'],
                            self.net.im_info: blobs['im_info'],
                            self.net.keep_prob: 0.5,
                            self.net.gt_boxes_bv: blobs['gt_boxes_bv'],
                            self.net.gt_boxes_3d: blobs['gt_boxes_3d'],
                            self.net.gt_boxes_corners: blobs['gt_boxes_corners'],
                            self.net.calib: blobs['calib']}
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
                            print 'Valid step: {:d}/{:d} , rpn recall = {:.3f}'\
                                  .format(data_idx + 1,self.val_epoch,float(recalls_[1]) / recalls_[2])
                            print('    class bg precision = {:.3f}  recall = {:.3f}'.format(
                                (one_hist[0, 0] / (one_hist[0, 0] + one_hist[1, 0])),
                                (one_hist[0, 0] / (one_hist[0, 0] + one_hist[0, 1]))))
                            print('    class car precision = {:.3f}  recall = {:.3f}'.format(
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[0, 1])),
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[1, 0]))))

                precise_total = hist[1, 1] / (hist[1, 1] + hist[0, 1])
                recall_total = hist[1, 1] / (hist[1, 1] + hist[1, 0])
                recall_rpn = pred_tp_cnt / gt_cnt
                valid_summary = tf.summary.merge([rpn_recall_smy_op, cubic_recall_smy_op, cubic_prec_smy_op])
                valid_res = sess.run(valid_summary, feed_dict={epoch_rpn_recall: recall_rpn,
                                                               epoch_cubic_recall: recall_total,
                                                               epoch_cubic_precise: precise_total})
                train_writer.add_summary(valid_res, epo_cnt + 1)
                print 'Validation of epoch_{}: rpn_recall {:.3f} cubic_precision = {:.3f}  cubic_recall = {:.3f}'\
                      .format(epo_cnt + 1,recall_rpn,precise_total,recall_total)
        self.snapshot(sess, iter, final=True)
        print 'Training process has done, enjoy every day !'

def network_training_sti(network, data_set, args):
    net = CubicNet_Train_sti(network, data_set, args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(cfg.LOG_DIR, sess.graph, max_queue=300)
        net.training(sess, train_writer)
