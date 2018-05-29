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
from tools.data_visualize import pcd_vispy,vispy_init,BoxAry_Theta,pcd_vispy_client
##================================================
# from multiprocessing import Process,Queue
# MSG_QUEUE = Queue(200)
##================================================
DEBUG = False
class msg_qt(object):
    def __init__(self,scans=None, img=None,queue=None, boxes=None, name=None,
                 index=0, vis_size=(800, 600), save_img=False,visible=True, no_gt=False):
        self.scans=scans,
        self.img=img,
        self.boxes=boxes,
        self.name=name,
        self.index=index,
        self.vis_size=vis_size,
        self.save_img=save_img,
        self.visible=visible,
        self.no_gt=no_gt,
        self.queue=queue

    def check(self):
        pass

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

        filename = os.path.join(output_dir, 'CubicNet_iter_{:d}'.format(iter) + '.ckpt')
        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

    @staticmethod
    def Rnet_modified_smooth_l1(sigma, bbox_pred, bbox_targets):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        diffs = tf.abs(tf.subtract(bbox_pred, bbox_targets))
        #
        over = tf.greater(diffs, np.pi/2) #TODO check
        diffs =tf.abs(tf.add(-diffs, tf.cast(over, dtype=tf.float32) * np.pi))

        smooth_l1_sign = tf.cast(tf.less(diffs, 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(diffs, diffs), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(diffs, 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = smooth_l1_result

        return outside_mul

    @staticmethod
    def angle_trans(angle):
        keep1 = tf.less_equal(angle, -np.pi/2)
        res1 = tf.add(angle, tf.cast(keep1, dtype=tf.float32) * np.pi)
        keep2 = tf.greater(res1, np.pi/2)
        res2 = tf.add(res1, -tf.cast(keep2, dtype=tf.float32) * np.pi)
        return res2

    def training(self, sess, train_writer):
        with tf.name_scope('loss_function'):
            RNet_rpn_yaw_pred = self.net.get_output('RNet_theta')[1]
            RNet_rpn_yaw_gt_delta = self.net.get_output('cubic_grid')[1]
            RNet_rpn_yaw_gt = self.net.get_output('rpn_rois')[1][:,-1]#rpn_3d_boxes:(x1,y1,z1),(x2,y2,z2),score,rpn_cls_label,yaw
            RNet_rpn_yaw_gt_new = RNet_rpn_yaw_gt-RNet_rpn_yaw_gt_delta
            RNet_rpn_yaw_pred_toshow = RNet_rpn_yaw_pred+RNet_rpn_yaw_gt_delta
            rpn_cls_labels = self.net.get_output('rpn_rois')[1][:,-2]#rpn_3d_boxes:(x1,y1,z1),(x2,y2,z2),score,rpn_cls_label,yaw

            RNet_rpn_yaw_pred = self.angle_trans(RNet_rpn_yaw_pred)
            RNet_rpn_yaw_gt_new = self.angle_trans(RNet_rpn_yaw_gt_new)

            debug_pred = tf.multiply(rpn_cls_labels,self.angle_trans(RNet_rpn_yaw_pred))
            debug_gt = tf.multiply(rpn_cls_labels,self.angle_trans(RNet_rpn_yaw_gt_new))

            tower_l1_loss = self.Rnet_modified_smooth_l1(sigma=3, bbox_pred=RNet_rpn_yaw_pred, bbox_targets=RNet_rpn_yaw_gt_new)
            tower_l1_loss_keep_positive = tf.multiply(rpn_cls_labels, tower_l1_loss)
            loss = tf.reduce_sum(tower_l1_loss_keep_positive)/(1e-5+tf.reduce_sum(tf.cast(tf.not_equal(tower_l1_loss_keep_positive, 0.0), dtype=tf.float32)))

        with tf.name_scope('train_op'):
            global_step = tf.Variable(1, trainable=False, name='Global_Step')
            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, 10000, 0.90, name='decay-Lr')
            Optimizer = tf.train.AdamOptimizer(lr)
            var_and_grad = Optimizer.compute_gradients(loss,var_list=tf.trainable_variables())
            train_op = Optimizer.minimize(loss, global_step=global_step)

        with tf.name_scope('debug_board'):
            tf.summary.scalar('total_loss', loss)
            glb_var = tf.trainable_variables()
            for i in range(len(glb_var)):
                tf.summary.histogram(glb_var[i].name, glb_var[i])
            tf.summary.image('theta', self.net.get_output('RNet_theta')[0],max_outputs=50)
            merged = tf.summary.merge_all() #hxd: before the next summary ops

        with tf.name_scope('epoch_valid'):
            epoch_cube_theta = tf.placeholder(dtype=tf.float32)
            epoch_cube_theta_sum_op = tf.summary.scalar('valid_los', epoch_cube_theta)

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
        print 'Variables to train: ',trainable_var_for_chk

        timer = Timer()
        rpn_rois_3d = self.net.get_output('rpn_rois')[1]

        if DEBUG:
            pass # TODO: Essential step(before sess.run) for using vispy beacuse of the bug of opengl or tensorflow
            vispy_init()
        i=0
        training_series = range(self.epoch)  #self.epoch
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
                    self.net.calib: blobs['calib'],
                }

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                timer.tic()
                debug_pred_,delta_,RNet_rpn_yaw_gt_delta_,rpn_rois_3d_,loss_,RNet_rpn_yaw_pred_toshow_,debug_gt_,merged_,_ = \
                    sess.run([debug_pred,tower_l1_loss_keep_positive,RNet_rpn_yaw_gt_delta,rpn_rois_3d,loss,RNet_rpn_yaw_pred_toshow,debug_gt,merged,train_op,]
                             ,feed_dict=feed_dict,options=run_options, run_metadata=run_metadata)
                # debug_pred_,delta_,RNet_rpn_yaw_gt_delta_,rpn_rois_3d_,RNet_rpn_yaw_pred_toshow_,debug_gt_,merged_, = \
                #     sess.run([debug_pred,tower_l1_loss_keep_positive,RNet_rpn_yaw_gt_delta,rpn_rois_3d,RNet_rpn_yaw_pred_toshow,debug_gt,merged,]
                #              ,feed_dict=feed_dict,options=run_options, run_metadata=run_metadata)
                timer.toc()

                if iter % cfg.TRAIN.ITER_DISPLAY == 0:
                    print 'Iter: %d/%d, Serial_num: %s, Speed: %.3fs/iter, Loss: %.3f '%(iter,self.args.epoch_iters * self.epoch, blobs['serial_num'],timer.average_time,loss_)
                    print 'theta_delta:     ',
                    for i in range(50):
                        if delta_[i]!=0.0:
                            print '%6.3f' % (delta_[i]),
                    print '\nPredicted angle: ',
                    for j in range(50):
                        if debug_pred_[j]!=0.0:
                            print '%6.3f' % (debug_pred_[j]),
                    print '\nGt yaw angle:    ',
                    for j in range(50):
                        if debug_gt_[j]!=0.0:
                            print '%6.3f' % (debug_gt_[j]),
                    print '\n'
                if iter % 20 == 0 and cfg.TRAIN.TENSORBOARD:
                    train_writer.add_summary(merged_, iter)
                    pass
                if (iter % 4000==0 and cfg.TRAIN.DEBUG_TIMELINE) or (iter == 100):
                    #chrome://tracing
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(cfg.LOG_DIR+'/' +'training-step-'+ str(iter).zfill(7) + '.ctf.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                    trace_file.close()
                if DEBUG:
                    scan = blobs['lidar3d_data']
                    cubic_cls_value = np.ones([cfg.TRAIN.RPN_POST_NMS_TOP_N],dtype=np.float32)*0
                    boxes = BoxAry_Theta(gt_box3d=blobs['gt_boxes_3d'], pre_box3d=rpn_rois_3d_,pre_theta_value=RNet_rpn_yaw_pred_toshow_,pre_cube_cls=cubic_cls_value)# RNet_rpn_yaw_pred_toshow_  rpn_rois_3d_[:,-1]
                    pcd_vispy(scan, boxes=boxes,name='CubicNet training',index=i,vis_size=(800, 600),save_img=False,visible=False)
                    i+=1
            if cfg.TRAIN.EPOCH_MODEL_SAVE:#iter % 2000==0 and :
                self.snapshot(sess, iter)
                pass
            if cfg.TRAIN.USE_VALID and True:#TODO: to complete the valid process
                with tf.name_scope('valid_cubic_' + str(epo_cnt + 1)):
                    print 'Valid the net at the end of epoch_{} ...'.format(epo_cnt + 1)
                    valid_loss_total = 0.0
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
                            self.net.calib: blobs['calib'],
                        }
                        loss_valid = sess.run(loss, feed_dict=feed_dict_)
                        # train_writer.add_summary(valid, data_idx)

                        valid_loss_total += loss_valid
                        if cfg.TRAIN.VISUAL_VALID and data_idx % 20 == 0:
                            print 'Valid step: {:d}/{:d} , theta_loss = {:.3f}'\
                                  .format(data_idx + 1,self.val_epoch,float(loss_valid))

                        if data_idx % 20 ==0 and cfg.TRAIN.TENSORBOARD:
                            pass
                            # train_writer.add_summary(valid_result_, data_idx/20+epo_cnt*1000)

                valid_summary = tf.summary.merge([epoch_cube_theta_sum_op])
                valid_res = sess.run(valid_summary, feed_dict={epoch_cube_theta:float(valid_loss_total)/self.val_epoch})
                train_writer.add_summary(valid_res, epo_cnt + 1)
                print 'Validation of epoch_{}:theta_loss_total = {:.3f}\n'\
                      .format(epo_cnt + 1,float(valid_loss_total)/self.val_epoch)
            random.shuffle(training_series)  # shuffle the training series
        print 'Training process has done, enjoy every day !'



def network_training(network, data_set, args):
    net = CubicNet_Train(network, data_set, args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(cfg.LOG_DIR, sess.graph, max_queue=300)
        net.training(sess, train_writer)
