# coding=utf-8
import tensorflow as tf
import numpy as np
from network.config import cfg


class vfe_encoder(object):
    def __init__(self, out_channels, name):
        self.uint = out_channels/2
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            with tf.variable_scope('vfe_encoder_dense', reuse=tf.AUTO_REUSE) as scope:
                self.dense = tf.layers.Dense(self.uint, tf.nn.relu, _reuse=tf.AUTO_REUSE, _scope=scope,name='v')
            with tf.variable_scope('vfe_encoder_batch_norm', reuse=tf.AUTO_REUSE) as scope:
                self.batch_norm = tf.layers.BatchNormalization(fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs, training=True):
        out=self.dense.apply(inputs)
        pointwise = self.batch_norm.apply(out, training)
        aggregated = tf.reduce_max(pointwise, axis=0, keep_dims=True,name='vfe_encoder_aggregated')
        repeated = tf.tile(aggregated, [tf.shape(pointwise)[0], 1])
        concatenated = tf.concat([pointwise, repeated], axis=1,name='vfe_encoder_concatenated')

        return concatenated


class VFE(object):
    def __init__(self,shape,name='',trainable_=True):
        # assert len(shape)==3, 'Class VFE has wrong input shape:{}'.format(shape)
        self.net_shape = shape
        self.trainable = trainable_
        self.output = []  # convenient for debug in main()
        self.extract = []  # convenient for debug in main()
        self.vfe_feature = []
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            with tf.variable_scope('vef_encoder', reuse=tf.AUTO_REUSE) as scope:
                self.vfe_1 = vfe_encoder(self.net_shape[0], name='VFE-1')
                self.vfe_2 = vfe_encoder(self.net_shape[1], name='VFE-2')
            with tf.variable_scope('reduce_fc', reuse=tf.AUTO_REUSE) as scope:
                self.dense = tf.layers.Dense(self.net_shape[2], tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
                self.batch_norm = tf.layers.BatchNormalization(name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE,_scope=scope)
            with tf.variable_scope('extract_fc', reuse=tf.AUTO_REUSE) as scope:
                self.dense2 = tf.layers.Dense(self.net_shape[3], tf.nn.relu, name='dense2', _reuse=tf.AUTO_REUSE,_scope=scope)
            with tf.variable_scope('class_fc', reuse=tf.AUTO_REUSE) as scope:
                self.dense3 = tf.layers.Dense(self.net_shape[4],use_bias=False, kernel_initializer=None,name='dense3', _reuse=tf.AUTO_REUSE,_scope=scope)
            # TODO: little bug: when the batch of the input of BatchNormalization is 1,bn's output turn to all-zeros

    def get_vfe_feature(self,in_put):
        # TODO: replace python for loop with tf.where_loop
        for i in range(cfg.TRAIN.RPN_POST_NMS_TOP_N):
            indice = tf.where(tf.equal(in_put[:, 0], i))
            extract = tf.reshape(tf.gather(in_put, indice),[-1,8])[:,1:8]
            vfe_feature = tf.reshape(self.compute(extract),[-1,self.net_shape[2]])
            extract = self.dense2.apply(vfe_feature)
            res = self.dense3.apply(extract)
            self.vfe_feature.append(vfe_feature)
            self.extract.append(extract)
            self.output.append(res)
            # TODO: use tf.map_fn to accelerate the speed  seeing in voxel_net tf.map_fn(self.compute,self.input)
        return self.output,self.vfe_feature,self.extract

    def compute(self,features):
        def fully_connected(in_data=None):
            in_data = self.vfe_1.apply(in_data, self.trainable)
            in_data = self.vfe_2.apply(in_data, self.trainable)
            in_data = self.dense.apply(in_data)
            in_data = self.batch_norm.apply(in_data, self.trainable)

            return tf.reduce_max(in_data, axis=0)

        def zeros_array(length=128):
            return np.zeros([1,length],dtype=np.float32)

        condition =tf.equal(tf.shape(features)[0],0)
        res = tf.cond(condition,lambda:zeros_array(self.net_shape[2]),lambda:fully_connected(features))

        return res

# class vfe_classifier(object):
#     def __init__(self,inputs,net_shape,name=None,trainable_=True):
#         self.in_data = inputs
#         self.shape=net_shape
#         self.trainable = trainable_
#         self.name = name
#         # with tf.variable_scope('asd', reuse=tf.AUTO_REUSE) as scope:
#         #     tf.add_to_collection('result_cbc', name)
#
#         loop_cnt = tf.constant(0,dtype=tf.float32)
#         res = tf.Variable(np.arange(14).reshape(2,7),dtype=tf.float32,expected_shape=[2,7])
#         self.outs=tf.while_loop(self.cond,self.body,[loop_cnt,res])
#         # self.outs = tf.get_collection('result_cbc')
#         pass
#
#     def cond(self,i,k):
#         condition = tf.less(i, cfg.TRAIN.RPN_POST_NMS_TOP_N)
#
#         return condition
#
#     def body(self,i,k):
#         with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
#             indice = tf.where(tf.equal(self.in_data[:, 0], i))
#             extract = tf.reshape(tf.gather(self.in_data, indice), [-1, 8])[:, 1:8]
#
#             condition = tf.equal(tf.shape(extract)[0], 0)
#             re_extract = tf.cond(condition, lambda: extract, lambda: tf.constant(0,dtype=tf.float32,shape=[1,7]))
#
#             regularizer = tf.contrib.layers.l2_regularizer(0.05)
#             with tf.variable_scope('vfe_1', reuse=tf.AUTO_REUSE) as scope:
#                 vfe_1 = tf.contrib.layers.fully_connected(re_extract,self.shape[0],weights_regularizer=regularizer,reuse=tf.AUTO_REUSE,scope=scope)
#                 vfe_1_bn = tf.contrib.layers.batch_norm(vfe_1,reuse=tf.AUTO_REUSE,scope=scope)
#             with tf.variable_scope('vfe_2', reuse=tf.AUTO_REUSE) as scope:
#                 vfe_2 = tf.contrib.layers.fully_connected(vfe_1_bn,self.shape[1],weights_regularizer=regularizer,reuse=tf.AUTO_REUSE,scope=scope)
#                 vfe_2_bn = tf.contrib.layers.batch_norm(vfe_2,reuse=tf.AUTO_REUSE,scope=scope)
#             with tf.variable_scope('vox_feature', reuse=tf.AUTO_REUSE) as scope:
#                 vfe_3 = tf.contrib.layers.fully_connected(vfe_2_bn,self.shape[2],weights_regularizer=regularizer,reuse=tf.AUTO_REUSE,scope=scope)
#                 vox_feature = tf.reshape(tf.reduce_max(vfe_3, axis=0),[-1,self.shape[2]])
#             with tf.variable_scope('ext_layer', reuse=tf.AUTO_REUSE) as scope:
#                 ext_layer = tf.contrib.layers.fully_connected(vox_feature,self.shape[3],weights_regularizer=regularizer,reuse=tf.AUTO_REUSE,scope=scope)
#                 # ext_layer_bn = tf.contrib.layers.batch_norm(ext_layer,reuse=tf.AUTO_REUSE,scope=scope)
#             with tf.variable_scope('cubic', reuse=tf.AUTO_REUSE) as scope:
#                 cubic_cls = tf.contrib.layers.fully_connected(ext_layer, self.shape[4], activation_fn=None,weights_regularizer=regularizer,reuse=tf.AUTO_REUSE, scope=scope)
#             # sdd = tf.constant(5, dtype=tf.float32,shape=[1,2])
#             # return_res = tf.concat([k,extract[0:1,:]],axis=0)[-50:,:]
#
#         return i+1,extract


if __name__ == '__main__':
    data = np.arange(700, dtype=np.float32).reshape(100, 7)
    header = np.array(range(50), dtype=np.float32).reshape(-1,1)
    header = np.hstack((header,header)).reshape(-1,1)
    input = np.hstack((header,data))
    dat = tf.placeholder(dtype=tf.float32,shape=[None,8])
    # xx = tf.get_collection('result_cbc')
    # VFE_classify = VFE(shape=[32,128,128,32,2],name='VFE_classify',trainable_=True)
    # x = VFE_classify.get_vfe_feature(dat)

    VFE_classify=VFE([32,128,128,32,2],name='VFE')
    x = VFE_classify.get_vfe_feature(dat)

    glo_bar = tf.global_variables()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(cfg.LOCAL_LOG_DIR, sess.graph, max_queue=300)
        sess.run(tf.global_variables_initializer())
        feed={dat:input}
        tf.summary.histogram('data',values=x[0])
        merge = tf.summary.merge_all()
        res1,summary = sess.run([x,merge],feed_dict=feed)
        train_writer.add_summary(summary, 1)
        print res1


