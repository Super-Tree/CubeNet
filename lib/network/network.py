import numpy as np
import tensorflow as tf

from config import cfg
from classify.VFE import VFE

from classify.rpn_classify import rpn_serial_extract_tf
from rpn.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from rpn.proposal_layer_tf import proposal_layer_3d as proposal_layer_py_3d
from rpn.proposal_layer_tf import proposal_layer_3d_STI,generate_rpn
from classify.rpn_3dcnn import cubic_rpn_grid_pyfc,cubic

from classify.vfe_layer import vfe_cube_Gen
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()
        self.channel = 0

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, saver, ignore_missing=False):
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
                            if not ignore_missing:
                                raise

    def load_weigths(self, data_path, session, saver,specical_flag= False):
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
        except :
            from tensorflow.python import pywrap_tensorflow
            reader = pywrap_tensorflow.NewCheckpointReader(data_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:
                for key in var_to_shape_map:
                    try:
                        if specical_flag:
                            var = tf.get_variable('cubic_cnn/'+key, trainable=False)
                        else:
                            var = tf.get_variable(key, trainable=False)
                        session.run(var.assign(reader.get_tensor(key)))
                        print "    Assign pretrain model: " + key
                    except ValueError:
                        print "    Ignore variable:" + key

    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    # print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s' % layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s' % layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    def l2_regularizer(self, weight_decay=cfg.TRAIN.WEIGHT_DECAY, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                                 dtype=tensor.dtype.base_dtype,
                                                 name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')

        return regularizer

    @layer
    def concat_bn_relu(self, input, input2, phase_train=True, name='conv'):
        concat = tf.concat([input, input2], 3, name=name)
        concat_bn = self.b_n(concat, phase_train)
        concat_bn_relu = tf.nn.relu(concat_bn, name=name)
        return concat_bn_relu

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name='conv', relu=True, bn=False, phase_train=True, has_bias=True,
             padding=cfg.DEFAULT_PADDING, group=1,trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            kernel = self.make_var('weights', [k_h, k_w, c_i / group, c_o], init_weights, trainable)
            if has_bias:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
            else:
                biases = tf.constant(0.000, shape=[c_o])

            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)

            if bn:
                conv = self.b_n(conv, phase_train)

            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)

            return tf.nn.bias_add(conv, biases, name=scope.name)

    def b_n(self, conv, phase_train=True):
        conv = tf.cond(phase_train,
                       lambda: tf.contrib.layers.batch_norm(conv, decay=0.9, epsilon=1e-5, center=True, scale=True,
                                                            is_training=1, reuse=None, updates_collections=None),
                       lambda: tf.contrib.layers.batch_norm(conv, decay=0.9, epsilon=1e-5, center=True, scale=True,
                                                            is_training=0, reuse=None, updates_collections=None)
                       )
        return conv

    @layer
    def deconv(self, input, shape, c_o, ksize1=3, ksize2=3, stride1=2, stride2=2, name='upconv', biased=False,
               relu=True,padding=cfg.DEFAULT_PADDING,trainable=True):
        """ up-conv"""
        self.validate_padding(padding)

        c_in = input.get_shape()[3].value
        in_shape = tf.shape(input)
        if shape is None:
            # h = ((in_shape[1] - 1) * stride) + 1
            # w = ((in_shape[2] - 1) * stride) + 1
            h = ((in_shape[1]) * stride1)
            w = ((in_shape[2]) * stride2)
            new_shape = [in_shape[0], h, w, c_o]
        else:
            new_shape = [in_shape[0], shape[1], shape[2], c_o]
        output_shape = tf.stack(new_shape)

        filter_shape = [ksize1, ksize2, c_o, c_in]

        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            filters = self.make_var('weights', filter_shape, init_weights, trainable)
            # regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            deconv = tf.nn.conv2d_transpose(input, filters, output_shape,
                                            strides=[1, stride1, stride2, 1], padding=cfg.DEFAULT_PADDING,
                                            name=scope.name)
            # coz de-conv losses shape info, use reshape to re-gain shape
            deconv = tf.reshape(deconv, new_shape)

            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(deconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(deconv, biases)
            else:
                if relu:
                    return tf.nn.relu(deconv)
                return deconv

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=cfg.DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=cfg.DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def proposal_transform(self, input, name, target='bv'):
        """ transform 3d propasal to different view """
        assert (target in ('bv', 'img', 'fv'))
        if isinstance(input, tuple):
            input_bv = input[0]
            input_img = input[1]

        if target == 'bv':

            with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
                lidar_bv = input_bv
            return lidar_bv

        elif target == 'img':

            with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
                image_proposal = input_img
            return image_proposal

        elif target == 'fv':

            return None

    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)

        return tf.reshape(input,
                          [input_shape[0],
                           input_shape[1],
                           -1,
                           int(d)])

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def element_wise_mean(self, input):
        return None

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable,
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                              [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)
        else:
            return tf.nn.softmax(input, name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    @layer
    def anchor_target_layer(self, input, _feat_stride, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        # gt_boxes_bv = lidar_to_top(input[1])
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            rpn_labels, rpn_bbox_targets, rpn_rois_bv, rpn_rois_3d = tf.py_func(anchor_target_layer_py,
                                                                                [input[0], input[1], input[2], input[3],
                                                                                 _feat_stride],
                                                                                [tf.float32, tf.float32, tf.float32,
                                                                                 tf.float32])
            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')
            rpn_rois_bv = tf.reshape(rpn_rois_bv, [-1, 5], name='rpn_rois_bv')
            rpn_rois_3d = tf.reshape(rpn_rois_3d, [-1, 7], name='rpn_rois_3d')
            return rpn_labels, rpn_bbox_targets, rpn_rois_bv, rpn_rois_3d

    @layer
    def rpn_extraction(self, input, name):
        # if isinstance(input[0], tuple):
        #     input[0] = input[0][0]
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            rpn_serial_points = tf.py_func(rpn_serial_extract_tf, [input[0], input[1][1], input[2]], tf.float32)

        rpn_serial_points=tf.convert_to_tensor(rpn_serial_points, dtype=tf.float32)
        return rpn_serial_points

    @layer
    def rpn_points_classify(self, input, name):
        input = tf.reshape(input,[-1,8])
        classifier = VFE([32, 128, 128,32,2],name,trainable_=True)
        feature_stack,ad,bc= classifier.get_vfe_feature(input)
        return feature_stack,ad,bc

    @layer
    def proposal_layer_3d(self, input, _feat_stride, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            rpn_rois_bv, rpn_rois_3d,rpn_recall = tf.py_func(proposal_layer_py_3d,
                                                  [input[0], input[1], input[2], input[3], cfg_key, _feat_stride],
                                                  [tf.float32, tf.float32,tf.float32])
            rpn_rois_bv = tf.reshape(rpn_rois_bv, [-1, 7], name='rpn_rois_bv') # (x1,y1),(x2,y2),score,rpn_cls_label,yaw_gt
            rpn_rois_3d = tf.reshape(rpn_rois_3d, [-1, 9], name='rpn_rois_3d') # (x1,y1,z1),(l,w,h),score,rpn_cls_label,yaw_gt
        return rpn_rois_bv, rpn_rois_3d, rpn_recall

    @layer
    def proposal_layer_3d_STI(self, input, bounding,num,name):

        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            rpn_rois_3d= tf.py_func(proposal_layer_3d_STI,[input,bounding,num],[tf.float32])
            rpn_rois_3d = tf.reshape(rpn_rois_3d, [-1, 8], name='rpn_rois_3d')

        return rpn_rois_3d

    @layer
    def generate_rpn(self, input, _feat_stride, cfg_key, name):

        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            rpn_rois_bv, rpn_rois_3d = tf.py_func(generate_rpn,[input[0], input[1], input[2], cfg_key, _feat_stride],
                                                  [tf.float32, tf.float32])
            rpn_rois_bv = tf.reshape(rpn_rois_bv, [-1, 6], name='rpn_rois_bv') # [x,y,l,w,score,rpn_label=0]
            rpn_rois_3d = tf.reshape(rpn_rois_3d, [-1, 8], name='rpn_rois_3d') # [x,y,z,l,w,h,score,rpn_label=0]
        return rpn_rois_bv, rpn_rois_3d

    @layer
    def cubic_grid(self, input,method, name):
        lidar_points = input[0]
        rpn_3d_boxes = input[1][1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            stack_cubic, rpn_yaw_delta = tf.py_func(cubic_rpn_grid_pyfc, [lidar_points, rpn_3d_boxes, method], [tf.float32,tf.float32])
        # rpn_yaw_delta: The delta yaw cause by data augmentation and its value is generated randomly in Function"cubic_rpn_grid_pyfc"
        # and it is used for yaw regression in later network ,and add to origin yaw gt groundtruth
        return stack_cubic, rpn_yaw_delta#

    @layer
    def RNet_theta(self, input,name):
        input = input[0]# input has two elements,stack_cubic,rpn_new_yaw
        with tf.variable_scope("cubic_theta", reuse=tf.AUTO_REUSE) as scope:
            #T:[B,30,30,15,2] ->[B,30,30]
            bi_bv = tf.reduce_max(input[:,:,:,:,0],axis=3,keep_dims=True)
            layer = tf.reshape(bi_bv,[cfg.TRAIN.RPN_POST_NMS_TOP_N,30,30,1])
            layer = tf.layers.conv2d(layer, filters=32,kernel_size=5,strides=[1, 1],padding="same",activation=tf.nn.relu,name=None,use_bias=False)
            # layer = tf.layers.batch_normalization(layer)
            layer = tf.layers.max_pooling2d(layer,[2,2],[2,2])
            layer = tf.layers.conv2d(layer, filters=64,kernel_size=5,strides=[1, 1],padding="same",activation=tf.nn.relu,name=None,use_bias=True)
            # layer = tf.layers.batch_normalization(layer)
            layer = tf.layers.max_pooling2d(layer, [2,2], [2,2])
            # layer = tf.layers.average_pooling2d(layer,[6,6],[6,6])
            layer = tf.reshape(layer, [cfg.TRAIN.RPN_POST_NMS_TOP_N,-1])
            # layer = tf.layers.batch_normalization(layer)
            layer = tf.layers.dense(layer,1024,use_bias=True)
            # layer = tf.nn.dropout(layer,0.6)#TODO: test remove
            layer = tf.nn.relu(layer)
            layer = tf.layers.dense(layer, 1, use_bias=True)
            layer = tf.reshape(layer,[-1])
            # layer = tf.layers.conv2d(layer,filters=32,kernel_size=3,strides=[1, 1],activation=tf.nn.relu,name=None)

        return bi_bv,layer

    @layer
    def cubic_cnn(self,input,channels, name):
        input = input[0]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            shape_feature = self.shape_extractor(input,channels,name='ShapeExtractor')
            cubic3dcnn = cubic(channels)
            result = cubic3dcnn.apply(shape_feature)
        return result

    @layer
    def vfe_feature_Gen(self, input, method, name):
        lidar_points = input[0]
        rpn_3d_boxes = input[1][1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            stack_cube = tf.py_func(vfe_cube_Gen, [lidar_points, rpn_3d_boxes, method], [tf.float32])
        return stack_cube

    def shape_extractor(self, inputs,channel,name):
        from tensorflow.python.ops import init_ops

        def converter_grad(op, grad):
            return grad * 25

        def converter_op(kernel_w):
            extractor_int_ = np.greater(kernel_w, 0.0).astype(np.float32)

            return extractor_int_

        def py_func(func, inp, Tout, stateful=True, name_=None, grad=None):
            # Need to generate a unique name to avoid duplicates:
            rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

            tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
            g = tf.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name}):
                return tf.py_func(func, inp, Tout, stateful=stateful, name=name_)

        def tf_extractor(x, name__=None):
            with tf.name_scope(name__, "shape_extractor", [x]) as _name:
                z = py_func(converter_op,
                            [x],
                            [tf.float32],
                            name_=_name,
                            grad=converter_grad)  # <-- here's the call to the gradient
                return z[0]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            extractor_weighs_float = tf.get_variable('extractor_float', shape=[3, 3, 3, 1, channel[0]],
                                                     initializer=init_ops.variance_scaling_initializer)
            extractor_int = tf_extractor(extractor_weighs_float, name__='extractor_int')
            res = tf.nn.conv3d(inputs, extractor_int, strides=[1, 1, 1, 1, 1], padding='SAME',
                               name='shape_feature')
            out = tf.reshape(res, [-1, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], cfg.CUBIC_SIZE[2], channel[0]])

        return out
