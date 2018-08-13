
from config import cfg
from network import Network
import tensorflow as tf


_feat_stride = [8, 8]  # scale cnt of pool and stride
trainable_auto = False  # control the head network whether to be trained in cubic net

class train_net(Network):
    def __init__(self, args,net_channel=None):
        self.inputs = []
        self.channel = net_channel
        self.lidar3d_data = tf.placeholder(tf.float32, shape=[None, 4])
        self.lidar_bv_data = tf.placeholder(tf.float32, shape=[None, 601, 601, 9])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes_bv = tf.placeholder(tf.float32, shape=[None, 6])
        self.gt_boxes_3d = tf.placeholder(tf.float32, shape=[None, 8])
        self.gt_boxes_corners = tf.placeholder(tf.float32, shape=[None, 25])
        self.calib = tf.placeholder(tf.float32, shape=[None, 12])
        self.keep_prob = tf.placeholder(tf.float32)
        self.target_data = 0
        self.label_data = 0
        self.layers = dict({'lidar3d_data': self.lidar3d_data,
                            'lidar_bv_data': self.lidar_bv_data,
                            'calib': self.calib,
                            'im_info': self.im_info,
                            'gt_boxes_bv': self.gt_boxes_bv,
                            'gt_boxes_3d': self.gt_boxes_3d,
                            'gt_boxes_corners': self.gt_boxes_corners})
        self.setup(args)

    def setup(self, args):
        # for idx, dev in enumerate(gpu_id):
        #     with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)):
        (self.feed('lidar_bv_data')
         .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=trainable_auto)
         .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=trainable_auto)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=trainable_auto)
         .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=trainable_auto)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1', trainable=trainable_auto)
         .conv(3, 3, 256, 1, 1, name='conv3_2', trainable=trainable_auto)
         .conv(3, 3, 256, 1, 1, name='conv3_3', trainable=trainable_auto)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1', trainable=trainable_auto)
         .conv(3, 3, 512, 1, 1, name='conv4_2', trainable=trainable_auto)
         .conv(3, 3, 512, 1, 1, name='conv4_3', trainable=trainable_auto)
         .conv(3, 3, 512, 1, 1, name='conv5_1', trainable=trainable_auto)
         .conv(3, 3, 512, 1, 1, name='conv5_2', trainable=trainable_auto)
         .conv(3, 3, 512, 1, 1, name='conv5_3', trainable=trainable_auto))
        # ========= RPN ============
        (self.feed('conv5_3')
         # .deconv(shape=None, c_o=512, stride=2, ksize=3,  name='deconv_2x_1')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3', trainable=trainable_auto)
         .conv(1, 1, cfg.ANCHOR_CNT * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score', trainable=trainable_auto))
        (self.feed('rpn_conv/3x3')
         .conv(1, 1, cfg.ANCHOR_CNT * 3, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred', trainable=trainable_auto))

        (self.feed('rpn_cls_score', 'gt_boxes_bv', 'gt_boxes_3d', 'im_info')
         .anchor_target_layer(_feat_stride, name='rpn_anchors_label'))

        (self.feed('rpn_cls_score')
         .reshape_layer(2, name='rpn_cls_score_reshape')
         .softmax(name='rpn_cls_prob')
         .reshape_layer(cfg.ANCHOR_CNT * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info', 'gt_boxes_bv')
         .proposal_layer_3d(_feat_stride, 'TRAIN', name='rpn_rois'))

        # (self.feed('lidar3d_data','rpn_rois')
        #  .cubic_grid(method=args.method,name='cubic_grid')
        #  .RNet_theta(name='RNet_theta') # inference the yaw of rpn proposal
        #  )

        (self.feed('lidar3d_data', 'rpn_rois')
         .cubic_grid(method=args.method, name='cubic_grid')
         .cubic_cnn(channels=self.channel,name='cubic_cnn')
         )
        self.target_data = self.layers['cubic_grid'][0]
        self.label_data = self.layers['rpn_rois'][0][:, -2]

        # (self.feed('lidar3d_data', 'rpn_rois')
        #  .vfe_feature_Gen(method=args.method, name='cubic_grid')
        #  # .cubic_cnn(name='cubic_cnn')
        #  )
