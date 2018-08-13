# coding=utf-8
import os
import numpy as np
import tensorflow as tf
from tools.timer import Timer
from network.train_net import train_net
from network.config import cfg
from tools.data_visualize import vispy_init,pcd_vispy,boxary2dic,pcd_show_now
from dataset.dataset import dataset_KITTI_train
from easydict import EasyDict as edict

DEBUG = False

class BoxFactory(object):
    def __init__(self, arg):
        self.arg = arg
        self.dataset = dataset_KITTI_train(arg)
        self.net = train_net(arg,[32,64,128,128,64,2])
        self.result = self.net.target_data
        self.label = self.net.label_data
        self.saver = tf.train.Saver(max_to_keep=100)

    def processor(self, sess):
        sess.run(tf.global_variables_initializer())
        self.net.load_weigths(self.arg.weights, sess, self.saver)
        timer = Timer()
        vispy_init()
        positive_cnt = 0
        negative_cnt = 0
        data_use_for='train'
        if data_use_for=='valid':
            length=self.dataset.validing_rois_length
        elif data_use_for=='train':
            length = self.dataset.training_rois_length
        else:
            assert False,'There is something wrong in dataset description'

        for idx in range(length):
            blobs = self.dataset.get_minibatch(idx, data_use_for)
            feed_dict = {
                        self.net.lidar3d_data: blobs['lidar3d_data'],
                        self.net.lidar_bv_data: blobs['lidar_bv_data'],
                        self.net.im_info: blobs['im_info'],
                        self.net.keep_prob: 0.5,
                        self.net.gt_boxes_bv: blobs['gt_boxes_bv'],
                        self.net.gt_boxes_3d: blobs['gt_boxes_3d'],
                        self.net.gt_boxes_corners: blobs['gt_boxes_corners'],
                        self.net.calib: blobs['calib']}
            timer.tic()
            result_, label_ = sess.run([self.result, self.label], feed_dict=feed_dict)
            timer.toc()
            print('Begin to save data_cnt: ', idx)
            pos_p=os.path.join(self.arg.box_savepath,data_use_for,'POSITIVE')
            neg_p=os.path.join(self.arg.box_savepath,data_use_for,'NEGATIVE')
            if not os.path.exists(pos_p):
                os.makedirs(pos_p)
            if not os.path.exists(neg_p):
                os.makedirs(neg_p)

            for box_cnt in range(result_.shape[0]):
                box = result_[box_cnt].astype(np.int8)
                if label_[box_cnt]:
                    filename =os.path.join(pos_p,str(positive_cnt).zfill(6)+'.npy')
                    positive_cnt += 1
                else:
                    filename = os.path.join(neg_p, str(negative_cnt).zfill(6) + '.npy')
                    negative_cnt += 1
                np.save(filename, box)

def network_training():
    arg = edict()
    arg.imdb_type = 'kitti'
    arg.method = 'train'
    arg.box_savepath = '/home/likewise-open/SENSETIME/hexindong/DISK1/DATASET/KITTI/object/box_car_only'
    arg.weights = '/home/likewise-open/SENSETIME/hexindong/ProjectDL/cubic-local/MODEL_weights/RPN_MODEL_3/weights/CombiNet_iter_180000.ckpt'
    net = BoxFactory(arg)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        net.processor(sess)

def box_view(path, idx,size=0.1):
    import os
    filename = os.path.join(path, str(idx).zfill(6)+'.npy')
    data = np.load(filename)
    coordinate = np.array(np.where(data[:,:,:,0] == 1)).transpose(1,0)
    print('Points in cube is %d'%data.sum())
    coordinate -= [cfg.CUBIC_SIZE[0]/2,cfg.CUBIC_SIZE[1]/2,cfg.CUBIC_SIZE[2]/2]
    pcd_vispy(coordinate,boxes=boxary2dic(np.array([0,0,0,cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1],cfg.CUBIC_SIZE[2],0,0])),point_size=size)

def box_np_view(data1,data2 = None):
    stack = []
    data1 = data1.reshape([30, 30, 15, 1])
    coordinate1 = np.array(np.where(data1[:, :, :, 0] == 1)).transpose(1, 0)
    coordinate1 -= [cfg.CUBIC_SIZE[0]/2,cfg.CUBIC_SIZE[1]/2,cfg.CUBIC_SIZE[2]/2]
    if data2 is not None:
        data2 = data2.reshape([30, 30, 15, 1])
        coordinate2 = np.array(np.where(data2[:, :, :, 0] == 1)).transpose(1, 0)
        coordinate2 -= [cfg.CUBIC_SIZE[0]/2,cfg.CUBIC_SIZE[1]/2,cfg.CUBIC_SIZE[2]/2]

        stack.append(pcd_vispy(coordinate2,name='WINDOW:2',
                     boxes=boxary2dic(np.array([0,0,0,cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1],cfg.CUBIC_SIZE[2],0,0])),
                     point_size=0.1,
                     visible=False,
                     multi_vis=True))

    stack.append(pcd_vispy(coordinate1,name='WINDOW:1',
                           boxes=boxary2dic(np.array([0,0,0,cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1],cfg.CUBIC_SIZE[2],0,0])),
                           point_size=0.1,
                           visible=False,
                           multi_vis=True))

    pcd_show_now()


if __name__ == '__main__':
    # network_training()
    # path = '/home/likewise-open/SENSETIME/hexindong/DISK1/DATASET/KITTI/object/box_car_only/train/POSITIVE'
    path = '/home/likewise-open/SENSETIME/hexindong/DISK1/DATASET/KITTI/object/box_car_only/valid/POSITIVE'
    list_name=sorted(os.listdir(path))
    cnt_array=[]
    for idx in range(0,len(list_name),1):
        # print('BOX_ID:', idx)
        # box_view(path, idx,size=0.1)

        filename = os.path.join(path, str(idx).zfill(6) + '.npy')
        data = np.load(filename)
        cnt_array.append(data.sum())

    cnt_array_np=np.array(cnt_array,dtype=np.int32)
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print(cnt_array_np.max())
    # plt.style.use('ggplot')
    # 绘图：乘客年龄的频数直方图
    cnt_array_np[cnt_array_np>599]=599
    plt.hist(cnt_array_np,  # 绘图数据
             range=(0,600),
             bins=30,  # 指定直方图的条形数为20个
             color='steelblue',  # 指定填充色
             edgecolor='k',  # 指定直方图的边界色
             label='PtsNum of Cube')  # 为直方图呈现标签
    plt.tick_params(top='off', right='off')
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()