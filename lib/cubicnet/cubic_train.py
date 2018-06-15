# coding=utf-8
import random
import os
import math
import numpy as np
import tensorflow as tf
from tools.timer import Timer
from network.config import cfg
from tools.utils import fast_hist
from tensorflow.python.client import timeline
from tools.data_visualize import pcd_vispy, vispy_init, BoxAry_Theta

DEBUG = False
DEBUG_MEM = False

class CubicNet_Train(object):
    def __init__(self, network, data_set, args):
        self.saver = tf.train.Saver(max_to_keep=100)
        self.net = network
        self.dataset = data_set
        self.args = args
        self.random_folder = cfg.RANDOM_STR
        self.epoch = self.dataset.training_rois_length
        self.val_epoch = self.dataset.validing_rois_length

    def snapshot(self, sess, iter=None):
        output_dir = os.path.join(cfg.ROOT_DIR, 'output', self.random_folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = os.path.join(output_dir, 'CubicNet_epoch_{:d}'.format(iter) + '.ckpt')
        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

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
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
        outside_mul = smooth_l1_result

        return outside_mul

    def training(self, sess, train_writer):
        with tf.name_scope('loss_cubic'):
            rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score'), [-1, 2])
            rpn_label = tf.reshape(self.net.get_output('rpn_anchors_label')[0], [-1])

            rpn_keep = tf.where(tf.not_equal(rpn_label, -1))
            rpn_bbox_keep = tf.where(tf.equal(rpn_label, 1))  # only regression positive anchors

            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_keep), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_keep), [-1])

            cubic_cls_score = tf.reshape(self.net.get_output('cubic_cnn'), [-1, 2])
            cubic_cls_labels = tf.reshape(tf.cast(self.net.get_output('rpn_rois')[0][:, -2], tf.int64), [-1])

            if not cfg.TRAIN.FOCAL_LOSS:
                rpn_cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

                cubic_cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cubic_cls_score, labels=cubic_cls_labels))
            else:
                #### use as reference for pos&neg proposal balance
                # self.cls_loss = alpha * (
                #             -self.pos_equal_one * tf.log(self.p_pos + small_addon_for_BCE)) / self.pos_equal_one_sum \
                #                 + beta * (-self.neg_equal_one * tf.log(
                #     1 - self.p_pos + small_addon_for_BCE)) / self.neg_equal_one_sum
                # self.cls_loss = tf.reduce_sum(self.cls_loss)
                ####

                alpha = [1.0, 1.0]  # 0.25 for label=1
                gamma = 2
                rpn_cls_probability = tf.nn.softmax(rpn_cls_score)
                cubic_cls_probability = tf.nn.softmax(cubic_cls_score)

                # formula :  Focal Loss for Dense Object Detection: FL(p)= -((1-p)**gama)*log(p)
                rpn_cross_entropy = tf.reduce_mean(-tf.reduce_sum(
                    tf.one_hot(rpn_label, depth=2) * ((1 - rpn_cls_probability) ** gamma) * tf.log(
                        [cfg.EPS, cfg.EPS] + rpn_cls_probability), axis=1))

                cubic_cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(cubic_cls_labels, depth=2) * ((1 - cubic_cls_probability) ** gamma) * tf.log([cfg.EPS, cfg.EPS] + cubic_cls_probability)*alpha, axis=1))

            # bounding box regression L1 loss
            rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
            rpn_bbox_targets = self.net.get_output('rpn_anchors_label')[1]
            rpn_bbox_pred = tf.reshape(tf.gather(tf.reshape(rpn_bbox_pred, [-1, 3]), rpn_bbox_keep), [-1, 3])
            rpn_bbox_targets = tf.reshape(tf.gather(tf.reshape(rpn_bbox_targets, [-1, 3]), rpn_bbox_keep), [-1, 3])

            rpn_smooth_l1 = self.modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets)
            rpn_loss_box = tf.multiply(tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1])), 1.0)

            # loss = rpn_cross_entropy + rpn_loss_box + cubic_cross_entropy
            loss = cubic_cross_entropy

        with tf.name_scope('train_op'):
            global_step = tf.Variable(1, trainable=False, name='Global_Step')
            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, 10000, 0.90, name='decay-Lr')
            train_op = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(loss, global_step=global_step)

        with tf.name_scope('train_cubic'):
            tf.summary.scalar('total_loss', loss)
            # tf.summary.scalar('rpn_loss_box', rpn_loss_box)
            # tf.summary.scalar('rpn_cross_entropy', rpn_cross_entropy)
            # tf.summary.scalar('cubic_cross_entropy', cubic_cross_entropy)
            recall_RPN = 0.
            # bv_anchors = self.net.get_output('rpn_anchors_label')[2]
            # roi_bv = self.net.get_output('rpn_rois')[0] # (x1,y1),(x2,y2),score,label
            # data_bv = self.net.lidar_bv_data
            # data_gt = self.net.gt_boxes_bv # (x1,y1),(x2,y2),label
            # # gt_box = tf.concat([data_gt,data_gt[:, 4]], axis=1)
            # bbox = tf.concat([roi_bv,data_gt],axis=0)
            # image_rpn = tf.reshape(show_rpn_tf(data_bv, bbox), (1, 601, 601, -1))
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

            valid_summary = tf.summary.merge([rpn_recall_smy_op, cubic_recall_smy_op, cubic_prec_smy_op])

        sess.run(tf.global_variables_initializer())
        if self.args.fine_tune:
            print 'Loading pre-trained model weights from {:s}'.format(self.args.weights)
            self.net.load_weigths(self.args.weights, sess, self.saver)
            self.net.load_weigths(self.args.weights_cube, sess, self.saver,specical_flag=True)
        trainable_var_for_chk = tf.trainable_variables()
        print 'Variables to train: ', trainable_var_for_chk

        timer = Timer()
        rpn_rois = self.net.get_output('rpn_rois')
        cubic_grid = self.net.get_output('cubic_grid')
        cubic_cnn = self.net.get_output('cubic_cnn')

        if DEBUG:
            vispy_init()  # TODO: Essential step(before sess.run) for using vispy beacuse of the bug of opengl or tensorflow
            # station = pcd_vispy_client(MSG_QUEUE,title='Vision')
            # vision_qt = Process(target=station.get_thread_data, args=(MSG_QUEUE,))
            # vision_qt.start()
            # print 'Process vision_qt started ...'

        if DEBUG_MEM:
            import os
            import matplotlib.pyplot as plt
            res_stack = []
            res = os.popen('ps aux|grep python2.7').read().split('hexindo+')
            for i in range(len(res)):
                if '/home/hexindong/Videos/cubic-local/experiment/main.py' in res[i]:
                    pid = res[i].split()[0]
                    break

        training_series = range(self.epoch)  # self.epoch
        sess.graph.finalize()  # in case of modifying graph for memory leak
        for epo_cnt in range(self.args.epoch_iters):
            for data_idx in training_series:  # DO NOT EDIT the "training_series",for the latter shuffle
                iter = global_step.eval()  # function "minimize()"will increase global_step
                blobs = self.dataset.get_minibatch(data_idx, 'train')  # get one batch
                feed_dict = {
                    self.net.lidar3d_data: blobs['lidar3d_data'],
                    self.net.lidar_bv_data: blobs['lidar_bv_data'],
                    self.net.im_info: blobs['im_info'],
                    self.net.keep_prob: 0.5,
                    self.net.gt_boxes_bv: blobs['gt_boxes_bv'],
                    self.net.gt_boxes_3d: blobs['gt_boxes_3d'],
                    self.net.gt_boxes_corners: blobs['gt_boxes_corners'],
                    self.net.calib: blobs['calib']}
                timer .tic()
                cubic_cls_score_, cubic_cls_labels_, rpn_rois_, cubic_cnn_, cubic_grid_, loss_, merged_, _ = sess.run(
                    [cubic_cls_score, cubic_cls_labels, rpn_rois, cubic_cnn, cubic_grid,loss, merged, train_op],
                    feed_dict=feed_dict)
                timer.toc()

                recall_RPN = recall_RPN + rpn_rois_[2][0]
                cubic_result = cubic_cls_score_.argmax(axis=1)
                one_hist = fast_hist(cubic_cls_labels_, cubic_result)
                cubic_car_cls_prec = one_hist[1, 1] / (one_hist[1, 1] + one_hist[0, 1]+1e-5)
                cubic_car_cls_recall = one_hist[1, 1] / (one_hist[1, 1] + one_hist[1, 0]+1e-5)

                # print('Step: ',iter)
                if iter % 100 == 0 and DEBUG_MEM:
                    with open('/proc/' + pid + '/statm', 'r') as f:
                        occupy_mb = int(f.readline().split()[1]) / 256.0
                        # res_stack.append(occupy_mb)

                    # if iter % 300 == 0:
                    #     plt.ylabel('Memory Occupy :MB')
                    #     plt.plot(res_stack)
                    #     plt.show()

                if iter % cfg.TRAIN.ITER_DISPLAY == 0:
                    print 'Iter: %d/%d, Serial_num: %s, speed: %.3fs/iter, loss: %.3f, rpn_recall: %.3f, cubic classify precise: %.3f,recall: %.3f' % \
                          (iter,self.args.epoch_iters * self.epoch, blobs['serial_num'],timer.average_time,loss_,recall_RPN / cfg.TRAIN.ITER_DISPLAY,cubic_car_cls_prec,cubic_car_cls_recall)
                    recall_RPN = 0.
                    print 'divine: ', str(cubic_result).translate(None,'\n')
                    print 'labels: ', str(cubic_cls_labels_).translate(None,'\n'), '\n'
                if iter % 40 == 0 and cfg.TRAIN.TENSORBOARD:
                    train_writer.add_summary(merged_, iter)
                    pass
                if (iter % 4000 == 0 and cfg.TRAIN.DEBUG_TIMELINE) or iter == 200:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _ = sess.run([cubic_cls_score], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    # chrome://tracing
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(cfg.LOG_DIR+'/' + 'training-step-' + str(iter).zfill(7) + '.ctf.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                    trace_file.close()
                if DEBUG:
                    scan = blobs['lidar3d_data']
                    gt_box3d = blobs['gt_boxes_3d'][:, (0, 1, 2, 3, 4, 5, 6, 7)]
                    gt_box3d = np.hstack((gt_box3d,np.ones([gt_box3d.shape[0], 2])*4))
                    pred_boxes = rpn_rois_[1]
                    # pred_boxes = np.hstack((rpn_rois_[1],cubic_result.reshape(-1,1)*2))
                    # bbox = np.vstack((pred_boxes, gt_box3d))
                    pcd_vispy(scan, boxes=BoxAry_Theta(gt_box3d,pred_boxes,pre_cube_cls=cubic_result), name='CubicNet training')
            if cfg.TRAIN.EPOCH_MODEL_SAVE:
                self.snapshot(sess, epo_cnt)
                pass
            if cfg.TRAIN.USE_VALID:
                with tf.name_scope('valid_cubic_' + str(epo_cnt + 1)):
                    print 'Valid the net at the end of epoch_{} ...'.format(epo_cnt + 1)
                    # roi_bv = self.net.get_output('rpn_rois')[0]
                    # cubu_bv = np.hstack((roi_bv,cubic_cls_labels.reshape(-1,1)))
                    # pred_rpn_ = show_rpn_tf(self.net.lidar_bv_data,cubu_bv)
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
                        cubic_cls_score_, cubic_cls_labels_, recalls_ = sess.run([cubic_cls_score, cubic_cls_labels, recalls], feed_dict=feed_dict_)
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
                                (one_hist[0, 0] / (one_hist[0, 0] + one_hist[1, 0]+1e-6)),
                                (one_hist[0, 0] / (one_hist[0, 0] + one_hist[0, 1]+1e-6))))
                            print('    class car precision = {:.3f}  recall = {:.3f}'.format(
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[0, 1]+1e-6)),
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[1, 0]+1e-6))))
                        if data_idx % 20 == 0 and cfg.TRAIN.TENSORBOARD:
                            pass
                            # train_writer.add_summary(valid_result_, data_idx/20+epo_cnt*1000)

                precise_total = hist[1, 1] / (hist[1, 1] + hist[0, 1]+1e-6)
                recall_total = hist[1, 1] / (hist[1, 1] + hist[1, 0]+1e-6)
                recall_rpn = pred_tp_cnt / gt_cnt
                valid_res = sess.run(valid_summary, feed_dict={epoch_rpn_recall: recall_rpn,
                                                               epoch_cubic_recall: recall_total,
                                                               epoch_cubic_precise: precise_total})
                train_writer.add_summary(valid_res, epo_cnt + 1)
                print 'Validation of epoch_{}: rpn_recall {:.3f} cubic_precision = {:.3f}  cubic_recall = {:.3f}'\
                      .format(epo_cnt + 1,recall_rpn,precise_total,recall_total)
            random.shuffle(training_series)  # shuffle the training series
        print 'Training process has done, enjoy every day !'

def network_training(network, data_set, args):
    net = CubicNet_Train(network, data_set, args)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(cfg.LOG_DIR, sess.graph, max_queue=300)
        net.training(sess, train_writer)