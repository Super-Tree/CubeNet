
import tensorflow as tf
from network.config import cfg
from tensorflow.python.client import timeline
from tools.timer import Timer
import numpy as np
from tools.data_visualize import pcd_vispy,vispy_init,test_show_rpn_tf

VISION_DEBUG = True

class network_testing_sti(object):
    def __init__(self, network, data_set, args):
        self.saver = tf.train.Saver(max_to_keep=100)
        self.net = network
        self.dataset = data_set
        self.args = args
        self.epoch = self.dataset.input_num

    def testing(self, sess, test_writer):
        with tf.name_scope('view_cubic_rpn'):
            roi_bv = self.net.get_output('rpn_rois')[0]
            data_bv = self.net.lidar_bv_data
            image_rpn = tf.reshape(test_show_rpn_tf(data_bv,roi_bv), (1, 601, 601, -1))
            tf.summary.image('lidar_bv_test', image_rpn)

            merged = tf.summary.merge_all()

        with tf.name_scope('load_weights'):
            weights = self.args.weights
            if weights.endswith('.ckpt'):
                print 'Loading test model weights from {:s}'.format(self.args.weights)
                self.saver.restore(sess, weights)
            else:
                print "error: Function [combinet_test.testing] can not load weights {:s}!".format(self.args.weights)
                return 0

        cubic_cls_score = tf.reshape(self.net.get_output('cubic_cnn'), [-1, 2])
        rpn_3d = tf.reshape(self.net.get_output('rpn_rois')[1],[-1,8])
        vispy_init()  # TODO: Essential step(before sess.run) for using vispy beacuse of the bug of opengl or tensorflow
        timer = Timer()
        for idx in range(self.epoch):
            blobs = self.dataset.get_minibatch(idx)
            feed_dict = {
                self.net.lidar3d_data: blobs['lidar3d_data'],
                self.net.lidar_bv_data: blobs['lidar_bv_data'],
                self.net.im_info: blobs['im_info'],
                self.net.calib: blobs['calib']}
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            timer.tic()
            cubic_cls_score_,rpn_3d_,summary = \
                sess.run([cubic_cls_score, rpn_3d, merged],
                         feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            timer.toc()
            cubic_result = cubic_cls_score_.argmax(axis=1)

            if idx % 3 ==0 and cfg.TEST.DEBUG_TIMELINE:
                # chrome://tracing
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(cfg.LOG_DIR + '/' +'testing-step-'+ str(idx).zfill(7) + '.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()
            if idx % cfg.TEST.ITER_DISPLAY == 0:
                pass
                print 'Test: %06d/%06d  speed: %.4f s / iter' % (idx+1, self.epoch, timer.average_time)
            if VISION_DEBUG:
                scan = blobs['lidar3d_data']
                img = blobs['image_data']
                pred_boxes = np.hstack((rpn_3d_, cubic_result.reshape(-1, 1)*2))
                pcd_vispy(scan,img, pred_boxes,no_gt=True,index=idx,
                          save_img=cfg.TEST.SAVE_IMAGE,
                          visible=True,
                          name='CubicNet testing')
            if idx % 1 == 0 and cfg.TEST.TENSORBOARD:
                test_writer.add_summary(summary, idx)
                pass
        print 'Testing process has done, happy every day !'


def network_testing_sti(network, data_set, args):
    net = network_testing_sti(network, data_set, args)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        test_writer = tf.summary.FileWriter(cfg.LOG_DIR, sess.graph, max_queue=300)
        net.testing(sess, test_writer)


