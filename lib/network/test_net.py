import tensorflow as tf
from network import Network
from config import cfg

_feat_stride = [8, 8]
class test_net(Network):
    def __init__(self,args,trainable=False):
        self.inputs = []
        self.lidar3d_data = tf.placeholder(tf.float32, shape=[None, 4])
        self.lidar_bv_data = tf.placeholder(tf.float32, shape=[None, 601, 601, 9])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        # self.calib = tf.placeholder(tf.float32, shape=[None, 12])
        self.layers = dict({'lidar3d_data': self.lidar3d_data,
                            'lidar_bv_data': self.lidar_bv_data,
                            # 'calib': self.calib,
                            'im_info': self.im_info}
                           )
        self.trainable = trainable
        self.setup(args)

    def setup(self,args):
        # for idx, dev in enumerate(gpu_id):
        #     with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)):
        (self.feed('lidar_bv_data')
         .conv(3, 3, 64, 1, 1, name='conv1_1',trainable=self.trainable)
         .conv(3, 3, 64, 1, 1, name='conv1_2',trainable=self.trainable)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1',trainable=self.trainable)
         .conv(3, 3, 128, 1, 1, name='conv2_2',trainable=self.trainable)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1',trainable=self.trainable)
         .conv(3, 3, 256, 1, 1, name='conv3_2',trainable=self.trainable)
         .conv(3, 3, 256, 1, 1, name='conv3_3',trainable=self.trainable)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1',trainable=self.trainable)
         .conv(3, 3, 512, 1, 1, name='conv4_2',trainable=self.trainable)
         .conv(3, 3, 512, 1, 1, name='conv4_3',trainable=self.trainable)
         .conv(3, 3, 512, 1, 1, name='conv5_1',trainable=self.trainable)
         .conv(3, 3, 512, 1, 1, name='conv5_2',trainable=self.trainable)
         .conv(3, 3, 512, 1, 1, name='conv5_3',trainable=self.trainable))
        # ========= RPN ============
        (self.feed('conv5_3')
         # .deconv(shape=None, c_o=512, stride=2, ksize=3,  name='deconv_2x_1')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3',trainable=self.trainable)
         .conv(1, 1, cfg.ANCHOR_CNT * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score',trainable=self.trainable))

        (self.feed('rpn_conv/3x3')
         .conv(1, 1, cfg.ANCHOR_CNT * 3, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred',trainable=self.trainable))

        (self.feed('rpn_cls_score')
         .reshape_layer(2, name='rpn_cls_score_reshape')
         .softmax(name='rpn_cls_prob')
         .reshape_layer(cfg.ANCHOR_CNT * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .generate_rpn(_feat_stride, 'TEST', name='rpn_rois'))

        # (self.feed('lidar3d_data','rpn_rois')
        #  .cubic_grid(method=args.method,name='cubic_grid')
        #  .RNet_theta(name='RNet_theta')
        #  )

        (self.feed('lidar3d_data','rpn_rois')
         .cubic_grid(method=args.method,name='cubic_grid')
         .cubic_cnn(name='cubic_cnn')
         )
