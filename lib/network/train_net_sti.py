
from config import cfg
from network import Network
import tensorflow as tf


_feat_stride = [8, 8]  # scale cnt of pool and stride
trainable_auto = False  # control the head network whether to be trained in cubic net

class train_net_sti(Network):
    def __init__(self, args):
        self.inputs = []
        self.lidar3d_data = tf.placeholder(tf.float32, shape=[None, 4])
        self.gt_boxes_3d = tf.placeholder(tf.float32, shape=[None, 8])
        self.layers = dict({'lidar3d_data': self.lidar3d_data,
                            'gt_boxes_3d': self.gt_boxes_3d,
                            })
        self.setup(args)

    def setup(self, args):
        # for idx, dev in enumerate(gpu_id):
        #     with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)):
        # ========= RPN ============
        (self.feed('gt_boxes_3d')
         .proposal_layer_3d_STI(bounding=cfg.DETECTION_RANGE,num=cfg.TRAIN.RPN_POST_NMS_TOP_N,name='rpn_rois'))
        # ========= CUBIC_NET ============
        (self.feed('lidar3d_data','rpn_rois')
         .cubic_grid(name='cubic_grid')
         .cubic_cnn(name='cubic_cnn')
         )

