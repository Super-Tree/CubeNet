
import cv2
import re
import os
import time
import random
import socket

import cPickle
import numpy as np
import scipy.sparse
import os.path as osp
from network.config import cfg
from os.path import join as path_add
from easydict import EasyDict as edict
from tools.transform import camera_to_lidar_cnr, computeCorners3D, lidar_3d_to_bv, lidar_cnr_to_3d, my_conner2bvbox
from tools.pcd_py_method.py_pcd import point_cloud as pcd2npScan
from tools.read_lidar import point_cloud_2_top
from tools.printer import blue
from tools.calibration import HangzhouCalib as calib
from tools.pcd_py_method.py_pcd import point_cloud as pcd_loader
class dataset_KITTI_train(object):  # read txt files one by one
    def __init__(self, arguments):
        self._type = arguments.imdb_type  # kitti or sti
        self._classes = ('__background__', 'Car')  # , 'Pedestrian', 'Cyclist')
        self.num_classes = len(self._classes)
        self._data_path = osp.join(cfg.DATA_DIR, 'training')  # data path
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
        self.inputIndex = self.get_fileIndex(self._data_path)
        self.input_num = len(self.inputIndex['train_index'])
        self.input_val_num = len(self.inputIndex['valid_index'])
        self.bounding_box = [0, 0, 601, 601]  # remove the gt_bv_box which out of image 600x600
        # TODO: add data augmentation such like flipped and rotated
        self.training_rois, self.validing_rois = self.load_roidb()
        self.training_rois_length = len(self.training_rois)
        self.validing_rois_length = len(self.validing_rois)
        pass
    def prepare_roidb(self, roidb, val_roidb):
        """Enrich the imdb's roidb by adding some derived quantities that
        are useful for training. This function precomputes the maximum
        overlap, taken over ground-truth boxes, between each ROI and
        each ground-truth box. The class with maximum overlap is also
        recorded.
        """
        #  sizes = [PIL.Image.open(imdb.image_path_at(i)).size
        #  for i in xrange(imdb.total_input)]
        # sizes = [np.load(imdb.lidar_path_at(i)).shape for i in xrange(imdb.total_input)]

        indice = lambda i, name: self.inputIndex[name][i]
        for i in xrange(self.input_num):
            if roidb[i]['boxes_corners'] is None:
                print 'boxes_corners not correct', self.lidar3d_path_at(indice(i, 'train_index'))
                continue
            roidb[i]['lidar3d_path'] = self.lidar3d_path_at(indice(i, 'train_index'))
            roidb[i]['lidar_bv_path'] = self.lidar_bv_path_at(indice(i, 'train_index'))
            # roidb[i]['width'] = sizes[1]#sizes[i][1]
            # roidb[i]['height'] = sizes[0]#sizes[i][0]
            roidb[i]['calib'] = self.calib_at(indice(i, 'train_index'))

            # need gt_overlaps as a dense array for argmax
            gt_overlaps = roidb[i]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)

            roidb[i]['max_classes'] = max_classes
            roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)

            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]

            assert all(max_classes[nonzero_inds] != 0)

        for i in xrange(self.input_val_num):
            if val_roidb[i]['boxes_corners'] is None:
                print 'boxes_corners not correct', self.image_path_at(indice(i, 'valid_index'))
                continue
            val_roidb[i]['lidar3d_path'] = self.lidar3d_path_at(indice(i, 'valid_index'))
            val_roidb[i]['lidar_bv_path'] = self.lidar_bv_path_at(indice(i, 'valid_index'))
            val_roidb[i]['calib'] = self.calib_at(indice(i, 'valid_index'))

            # need gt_overlaps as a dense array for argmax
            gt_overlaps = val_roidb[i]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)

            val_roidb[i]['max_classes'] = max_classes
            val_roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)

            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]

            assert all(max_classes[nonzero_inds] != 0)
        return roidb, val_roidb

    def filter_roidb(self, roidb):
        """Remove roidb entries that have no usable RoIs."""

        def is_valid(entry):
            # Valid images have:
            #   (1) At least one foreground RoI OR
            #   (2) At least one background RoI
            overlaps = entry['max_overlaps']
            # find boxes with sufficient overlap
            fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                               (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
            # image is only valid if such boxes exist
            valid = len(fg_inds) > 0 or len(bg_inds) > 0
            return valid

        num = len(roidb)

        filtered_roidb = [entry for entry in roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                           num, num_after)
        return filtered_roidb

    def load_roidb(self):
        if socket.gethostname()=='hexindong':
            cache_file = os.path.join(self._data_path, 'train_gt_roidb_LOCAL.pkl')
            val_cache_file = os.path.join(self._data_path, 'valid_gt_roidb_LOCAL.pkl')
        else:
            cache_file = os.path.join(self._data_path, 'train_gt_roidb_SERVER.pkl')
            val_cache_file = os.path.join(self._data_path, 'valid_gt_roidb_SERVER.pkl')
        if os.path.exists(cache_file) & os.path.exists(val_cache_file):
            print 'Dataset will be loaded from existing cache file'
            with open(cache_file, 'rb') as fid:
                train_roidb = cPickle.load(fid)
            print '  train gt roidb loaded from {}'.format(cache_file)
            with open(val_cache_file, 'rb') as fid:
                valid_roidb = cPickle.load(fid)
            print '  valid gt roidb loaded from {}'.format(val_cache_file)

            return train_roidb, valid_roidb

        print 'Prepare the kitti dataset(train:{}/valid:{}) for training, please wait ...'.format(self.input_num, self.input_val_num)
        train_roidb = [self.load_kitti_annotation(index) for index in self.inputIndex['train_index']]
        valid_roidb = [self.load_kitti_annotation(index) for index in self.inputIndex['valid_index']]

        # 'Func prepare_roidb()':this function add more elements to parameters so the changes will be save to parameters
        # https://www.cnblogs.com/cmnz/p/6927260.html
        train_roidb, valid_roidb = self.prepare_roidb(train_roidb, valid_roidb)

        train_roidb = self.filter_roidb(train_roidb)
        valid_roidb = self.filter_roidb(valid_roidb)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(train_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        with open(val_cache_file, 'wb') as fid:
            cPickle.dump(valid_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(val_cache_file)

        return train_roidb, valid_roidb

    def lidar3d_path_at(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # set the prefix
        prefix = 'velodyne'
        #  lidar3d_path = '$Faster-RCNN_TF/data/KITTI/object/training/velodyne/000000.bin'
        lidar3d_path = os.path.join(self._data_path, prefix, str(index).zfill(6) + '.bin')
        assert os.path.exists(lidar3d_path), 'Path does not exist: {}'.format(lidar3d_path)
        return lidar3d_path

    def lidar_bv_path_at(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        index = int(index)
        # set the prefix
        prefix = 'lidar_bv'
        # lidar_bv_path = '$Faster-RCNN_TF/data/KITTI/object/training/lidar_bv/000000.npy'
        lidar_bv_path = os.path.join(self._data_path, prefix, str(index).zfill(6) + '.npy')
        assert os.path.exists(lidar_bv_path), \
            'Path does not exist: {}'.format(lidar_bv_path)
        return lidar_bv_path

    def calib_at(self, index):
        """
        Return the calib sequence.
        """

        calib_ori = self.load_kitti_calib(index)
        calib = np.zeros((4, 12))
        calib[0, :] = calib_ori['P2'].reshape(12)
        calib[1, :] = calib_ori['P3'].reshape(12)
        calib[2, :9] = calib_ori['R0'].reshape(9)
        calib[3, :] = calib_ori['Tr_velo2cam'].reshape(12)

        return calib

    def get_fileIndex(self, data_path):
        assign_path = osp.join(self._data_path, 'image_assign')
        if not os.path.exists(assign_path):  #
            input_num = len(os.listdir(osp.join(data_path, 'velodyne')))
            test_index = []
            train_index = []
            temp_index = sorted(random.sample(range(input_num), int(input_num * 0.9990)))
            # generate test index
            for i in range(input_num):
                if i not in temp_index:
                    test_index.append(i)
            # generate valid index
            valid_index = sorted(random.sample(temp_index, int(input_num * 0.160)))
            # generate train index
            for k in temp_index:
                if k not in valid_index:
                    train_index.append(k)

            os.mkdir(assign_path)
            file = open(assign_path + '/train.txt', 'w')
            for j in range(len(train_index)):
                file.write(str(train_index[j]))
                file.write('\n')
            file.close()

            file = open(assign_path + '/test.txt', 'w')
            for j in range(len(test_index)):
                file.write(str(test_index[j]))
                file.write('\n')
            file.close()

            file = open(assign_path + '/valid.txt', 'w')
            for j in range(len(valid_index)):
                file.write(str(valid_index[j]))
                file.write('\n')
            file.close()
        else:
            train_index = np.loadtxt(assign_path + '/train.txt', dtype=int)
            # test_index = np.loadtxt(assign_path + '/test.txt', dtype=int)
            valid_index = np.loadtxt(assign_path + '/valid.txt', dtype=int)

        return dict({'train_index': train_index, 'valid_index': valid_index})

    def load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from txt file in the KITTI
        format.
        """
        filename = os.path.join(self._data_path, 'label_2', str(index).zfill(6) + '.txt')
        calib = self.load_kitti_calib(index)  # calib
        Tr = calib['Tr_velo2cam']
        # print 'Loading: {}'.format(filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
        num_objs = len(lines)
        translation = np.zeros((num_objs, 3), dtype=np.float32)
        rys = np.zeros((num_objs), dtype=np.float32)
        lwh = np.zeros((num_objs, 3), dtype=np.float32)
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        boxes_bv = np.zeros((num_objs, 4), dtype=np.float32)
        boxes3D = np.zeros((num_objs, 6), dtype=np.float32)
        boxes3D_lidar = np.zeros((num_objs, 6), dtype=np.float32)
        boxes3D_cam_cnr = np.zeros((num_objs, 24), dtype=np.float32)
        boxes3D_corners = np.zeros((num_objs, 24), dtype=np.float32)
        alphas = np.zeros((num_objs), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, len(self._classes)), dtype=np.float32)
        # Load object bounding boxes into a data frame.
        ix = -1
        for line in lines:
            obj = line.strip().split(' ')
            try:
                cls = self._class_to_ind[obj[0].strip()]
            except:
                continue
            # ignore objects with undetermined difficult level
            # level = self.get_obj_level(obj)
            # if level > 3:
            #     continue
            ix += 1
            # 0-based coordinates
            alpha = float(obj[3])
            x1 = float(obj[4])
            y1 = float(obj[5])
            x2 = float(obj[6])
            y2 = float(obj[7])
            h = float(obj[8])
            w = float(obj[9])
            l = float(obj[10])
            tx = float(obj[11])
            ty = float(obj[12])
            tz = float(obj[13])
            ry = float(obj[14])

            rys[ix] = ry
            lwh[ix, :] = [l, w, h]
            alphas[ix] = alpha
            translation[ix, :] = [tx, ty, tz]
            boxes[ix, :] = [x1, y1, x2, y2]
            boxes3D[ix, :] = [tx, ty, tz, l, w, h]
            # convert boxes3D cam to 8 corners(cam)
            boxes3D_cam_cnr_single = computeCorners3D(boxes3D[ix, :], ry)
            boxes3D_cam_cnr[ix, :] = boxes3D_cam_cnr_single.reshape(24)
            # convert 8 corners(cam) to 8 corners(lidar)
            boxes3D_corners[ix, :] = camera_to_lidar_cnr(boxes3D_cam_cnr_single, Tr)
            # convert 8 corners(lidar) to  lidar boxes3D
            boxes3D_lidar[ix, :] = lidar_cnr_to_3d(boxes3D_corners[ix, :], lwh[ix, :])
            # convert 8 corners(lidar) to lidar bird view
            # avb = lidar_3d_to_bv(boxes3D_lidar[ix, :])
            boxes_bv[ix, :] = my_conner2bvbox(boxes3D_corners[ix, :])  # boxes3D_corners.shape = [3,8]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

            if not self.check_box_bounding(boxes_bv[ix, :]):
                rys[ix] = 0
                lwh[ix, :] = [0, 0, 0]
                alphas[ix] = 0
                translation[ix, :] = [0, 0, 0]
                boxes[ix, :] = [0, 0, 0, 0]
                boxes3D[ix, :] = [0, 0, 0, 0, 0, 0]
                boxes3D_cam_cnr[ix, :] = np.zeros((24), dtype=np.float32)
                boxes3D_corners[ix, :] = np.zeros((24), dtype=np.float32)
                boxes3D_lidar[ix, :] = [0, 0, 0, 0, 0, 0]
                boxes_bv[ix, :] = [0, 0, 0, 0]
                gt_classes[ix] = 0
                overlaps[ix, cls] = 0
                ix = ix - 1

        # rys.resize(ix + 1)
        # lwh.resize(ix + 1, 3)
        # translation.resize(ix + 1, 3)
        # alphas.resize(ix + 1)
        # boxes.resize(ix + 1, 4)
        # boxes_bv.resize(ix + 1, 4)
        # boxes3D.resize(ix + 1, 6)
        # boxes3D_lidar.resize(ix + 1, 6)
        # boxes3D_cam_cnr.resize(ix + 1, 24)
        # boxes3D_corners.resize(ix + 1, 24)
        # gt_classes.resize(ix + 1)
        # overlaps.resize(ix + 1, self.num_classes)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        # if index == '000142':
        #     print(overlaps)

        return {'ry': rys,
                'lwh': lwh,
                'boxes': boxes,
                'boxes_bv': boxes_bv,
                'boxes_3D_cam': boxes3D,
                'boxes_3D': boxes3D_lidar,
                'boxes3D_cam_corners': boxes3D_cam_cnr,
                'boxes_corners': boxes3D_corners,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'xyz': translation,
                'alphas': alphas,
                'flipped': False}

    def load_kitti_calib(self, index):
        """
        load projection matrix
        """
        calib_dir = os.path.join(self._data_path, 'calib', str(index).zfill(6) + '.txt')
        #         P0 = np.zeros(12, dtype=np.float32)
        #         P1 = np.zeros(12, dtype=np.float32)
        #         P2 = np.zeros(12, dtype=np.float32)
        #         P3 = np.zeros(12, dtype=np.float32)
        #         R0 = np.zeros(9, dtype=np.float32)
        #         Tr_velo_to_cam = np.zeros(12, dtype=np.float32)
        #         Tr_imu_to_velo = np.zeros(12, dtype=np.float32)

        #         j = 0
        with open(calib_dir) as fi:
            lines = fi.readlines()
        # assert(len(lines) == 8)

        # obj = lines[0].strip().split(' ')[1:]
        # P0 = np.array(obj, dtype=np.float32)
        # obj = lines[1].strip().split(' ')[1:]
        # P1 = np.array(obj, dtype=np.float32)
        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        # obj = lines[6].strip().split(' ')[1:]
        # P0 = np.array(obj, dtype=np.float32)

        return {'P2': P2.reshape(3, 4),
                'P3': P3.reshape(3, 4),
                'R0': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

    def get_obj_level(self, obj):
        height = float(obj[7]) - float(obj[5]) + 1
        trucation = float(obj[1])
        occlusion = float(obj[2])
        if height >= 40 and trucation <= 0.15 and occlusion <= 0:
            return 1
        elif height >= 25 and trucation <= 0.3 and occlusion <= 1:
            return 2
        elif height >= 25 and trucation <= 0.5 and occlusion <= 2:
            return 3
        else:
            return 4

    def augmentation_of_data(self):
        # Rotation of the image or change the scale
        pass

    def get_minibatch(self, idx=0, name='train'):
        """Given a roidb, construct a minibatch sampled from it."""
        dataset = self.training_rois if name == 'train' else self.validing_rois

        im_scales = [1]
        lidar_bv = np.load(dataset[idx]['lidar_bv_path'])
        lidar_bv_blob = lidar_bv.reshape((1, lidar_bv.shape[0], lidar_bv.shape[1], lidar_bv.shape[2]))
        lidar3d = np.fromfile(dataset[idx]['lidar3d_path'], dtype=np.float32)
        lidar3d_blob = lidar3d.reshape((-1,4))

        gt_inds = np.where(dataset[idx]['gt_classes'] != 0)[0]
        gt_boxes_bv = np.empty((len(gt_inds), 6), dtype=np.float32)
        gt_boxes_bv[:, 0:4] = dataset[idx]['boxes_bv'][gt_inds, :]
        gt_boxes_bv[:, 4] = dataset[idx]['gt_classes'][gt_inds]
        gt_boxes_bv[:, 5] = dataset[idx]['ry'][gt_inds]

        # gt boxes 3d: (x, y, z, l, w, h, cls)
        gt_boxes_3d = np.empty((len(gt_inds), 8), dtype=np.float32)
        gt_boxes_3d[:, 0:6] = dataset[idx]['boxes_3D'][gt_inds, :]
        gt_boxes_3d[:, 6] = dataset[idx]['gt_classes'][gt_inds]

        thetas = dataset[idx]['ry'][gt_inds]
        out_1 = np.where(thetas < -np.pi/2)[0]
        thetas[out_1]+=np.pi
        out_2 = np.where(thetas > np.pi/2)[0]
        thetas[out_2]-=np.pi
        gt_boxes_3d[:, 7]= thetas#TODO:Check
        # if thetas<(-1.5708)
        #     thetas+=3.1415926
        # elif thetas>1.5708
        #     thetas = 3.1415926-thetas
        # gt_boxes_3d[:, 7]=thetas

        # gt boxes corners: (x0, ... x7, y0, y1, ... y7, z0, ... z7, cls)
        gt_boxes_corners = np.empty((len(gt_inds), 25), dtype=np.float32)
        gt_boxes_corners[:, 0:24] = dataset[idx]['boxes_corners'][gt_inds, :]
        gt_boxes_corners[:, 24] = dataset[idx]['gt_classes'][gt_inds]

        blobs = dict({'lidar_bv_data': lidar_bv_blob,
                      'lidar3d_data': lidar3d_blob,
                      'calib': dataset[idx]['calib'],
                      'gt_boxes_bv': gt_boxes_bv,
                      'gt_boxes_3d': gt_boxes_3d,
                      'gt_boxes_corners': gt_boxes_corners,
                      'im_info': np.array([[lidar_bv_blob.shape[1], lidar_bv_blob.shape[2], im_scales[0]]],dtype=np.float32),
                      'serial_num': dataset[idx]['lidar_bv_path'].split('/')[-1][0:6]
                      })

        return blobs

    def check_box_bounding(self, bbox):
        stand = self.bounding_box
        if (bbox[0] > (stand[0] + 1)) & (bbox[1] > (stand[1] + 1)) & (bbox[2] < (stand[2] - 1)) & (
            bbox[3] < (stand[3] - 1)):
            return True
        else:
            return False

class dataset_KITTI_test(object):  # read txt files one by one
    def __init__(self, arguments):
        self._type = arguments.imdb_type  # kitti or sti
        self._classes = ('__background__', 'Car')  # , 'Pedestrian', 'Cyclist')
        self.num_classes = len(self._classes)
        if arguments.use_demo:
            self._data_path = osp.join(cfg.DATA_DIR, arguments.DemoChoice)  # data path
            if arguments.DemoChoice == 'drive_0064':
                self.prefix_tmp = 10
            elif arguments.DemoChoice == 'drive_0015':
                self.prefix_tmp = 10
            else:
                self.prefix_tmp = 10
        else:
            self._data_path = osp.join(cfg.DATA_DIR, 'testing')  # data path

        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
        self.inputIndex = self.get_fileIndex(self._data_path)
        self.input_num = len(self.inputIndex['test_index'])

        print 'The kitti dataset(cnt:{}) is using for testing ...'.format(self.input_num)
        self.roidb = self.prepare_roidb()
        pass

    def prepare_roidb(self):
        roidb = [dict({}) for _ in range(self.input_num)]
        indice = lambda idx, name: self.inputIndex[name][idx]
        for i in xrange(self.input_num):
            roidb[i]['lidar3d_path'] = self.lidar3d_path_at(indice(i, 'test_index'))
            # roidb[i]['lidar_bv_path'] = self.lidar_bv_path_at(indice(i, 'test_index'))
            roidb[i]['image_path'] = self.image_path_at(indice(i, 'test_index'))
            # roidb[i]['calib'] = self.get_calib(indice(i, 'test_index'))
        return roidb

    def image_path_at(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # set the prefix
        prefix = 'image_2'
        # image_path = '$Faster-RCNN_TF/data/KITTI/object/training/image_2/000000.png'
        image_path = os.path.join(self._data_path, prefix, str(index).zfill(self.prefix_tmp) + '.png')
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def lidar3d_path_at(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # set the prefix
        prefix = 'velodyne'
        # image_path = '$Faster-RCNN_TF/data/KITTI/object/training/image_2/000000.png'
        lidar3d_path = os.path.join(self._data_path, prefix, str(index).zfill(self.prefix_tmp) + '.bin')
        assert os.path.exists(lidar3d_path), 'Path does not exist: {}'.format(lidar3d_path)
        return lidar3d_path

    def lidar_bv_path_at(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        index = int(index)
        # set the prefix
        prefix = 'lidar_bv'
        # lidar_bv_path = '$Faster-RCNN_TF/data/KITTI/object/training/lidar_bv/000000.npy'
        lidar_bv_path = os.path.join(self._data_path, prefix, str(index).zfill(self.prefix_tmp) + '.npy')
        assert os.path.exists(lidar_bv_path), \
            'Path does not exist: {}'.format(lidar_bv_path)
        return lidar_bv_path

    def get_calib(self, index):

        calib_dir = os.path.join(self._data_path, 'calib', str(index).zfill(self.prefix_tmp) + '.txt')

        with open(calib_dir) as fi:
            lines = fi.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)

        calib_ori= {'P2': P2.reshape(3, 4),
                    'P3': P3.reshape(3, 4),
                    'R0': R0.reshape(3, 3),
                    'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}
        # calib_ori = self.load_kitti_calib(index)
        calib = np.zeros((4, 12))
        calib[0, :] = calib_ori['P2'].reshape(12)
        calib[1, :] = calib_ori['P3'].reshape(12)
        calib[2, :9] = calib_ori['R0'].reshape(9)
        calib[3, :] = calib_ori['Tr_velo2cam'].reshape(12)

        return calib

    def get_fileIndex(self, data_path):
        length = len(os.listdir(osp.join(data_path, 'velodyne')))
        test_index = range(length)
        return dict({'test_index': test_index})

    def get_minibatch(self, idx=0):
        """Given a roidb, construct a minibatch sampled from it."""
        dataset = self.roidb
        im_scales = [1]
        lidar3d = np.fromfile(dataset[idx]['lidar3d_path'], dtype=np.float32)
        lidar3d_blob = lidar3d.reshape((-1,4))

        lidar_bv = point_cloud_2_top(lidar3d_blob)
        lidar_bv_blob = lidar_bv.reshape((1, lidar_bv.shape[0], lidar_bv.shape[1], lidar_bv.shape[2]))

        img = cv2.imread(dataset[idx]['image_path'])

        blobs = dict({'lidar_bv_data': lidar_bv_blob,
                      'lidar3d_data': lidar3d_blob,
                      # 'calib': dataset[idx]['calib'],
                      'im_info': np.array([[lidar_bv_blob.shape[1], lidar_bv_blob.shape[2], im_scales[0]]],dtype=np.float32),
                      'image_data': img
                      })

        return blobs

class dataset_HANGZHOU_train(object):  # read txt files one by one
    def __init__(self, arguments):
        self._type = arguments.imdb_type  # kitti or sti
        self._classes = ('__background__', 'Car')  # , 'Pedestrian', 'Cyclist')
        self.num_classes = len(self._classes)
        self._data_path = osp.join(cfg.DATA_DIR, 'training')  # data path
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
        self.inputIndex = self.get_fileIndex(self._data_path)
        self.input_num = len(self.inputIndex['train_index'])
        self.input_val_num = len(self.inputIndex['valid_index'])
        self.bounding_box = [0, 0, 601, 601]  # remove the gt_bv_box which out of image 600x600
        # TODO: add data augmentation such like flipped and rotated
        self.training_rois, self.validing_rois = self.load_roidb()
        self.training_rois_length = len(self.training_rois)
        self.validing_rois_length = len(self.validing_rois)
        pass
    def prepare_roidb(self, roidb, val_roidb):
        """Enrich the imdb's roidb by adding some derived quantities that
        are useful for training. This function precomputes the maximum
        overlap, taken over ground-truth boxes, between each ROI and
        each ground-truth box. The class with maximum overlap is also
        recorded.
        """
        #  sizes = [PIL.Image.open(imdb.image_path_at(i)).size
        #  for i in xrange(imdb.total_input)]
        # sizes = [np.load(imdb.lidar_path_at(i)).shape for i in xrange(imdb.total_input)]

        indice = lambda i, name: self.inputIndex[name][i]
        for i in xrange(self.input_num):
            if roidb[i]['boxes_corners'] is None:
                print 'boxes_corners not correct', self.lidar3d_path_at(indice(i, 'train_index'))
                continue
            roidb[i]['lidar3d_path'] = self.lidar3d_path_at(indice(i, 'train_index'))
            roidb[i]['lidar_bv_path'] = self.lidar_bv_path_at(indice(i, 'train_index'))
            # roidb[i]['width'] = sizes[1]#sizes[i][1]
            # roidb[i]['height'] = sizes[0]#sizes[i][0]
            roidb[i]['calib'] = self.calib_at(indice(i, 'train_index'))

            # need gt_overlaps as a dense array for argmax
            gt_overlaps = roidb[i]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)

            roidb[i]['max_classes'] = max_classes
            roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)

            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]

            assert all(max_classes[nonzero_inds] != 0)

        for i in xrange(self.input_val_num):
            if val_roidb[i]['boxes_corners'] is None:
                print 'boxes_corners not correct', self.image_path_at(indice(i, 'valid_index'))
                continue
            val_roidb[i]['lidar3d_path'] = self.lidar3d_path_at(indice(i, 'valid_index'))
            val_roidb[i]['lidar_bv_path'] = self.lidar_bv_path_at(indice(i, 'valid_index'))
            val_roidb[i]['calib'] = self.calib_at(indice(i, 'valid_index'))

            # need gt_overlaps as a dense array for argmax
            gt_overlaps = val_roidb[i]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)

            val_roidb[i]['max_classes'] = max_classes
            val_roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)

            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]

            assert all(max_classes[nonzero_inds] != 0)
        return roidb, val_roidb

    def filter_roidb(self, roidb):
        """Remove roidb entries that have no usable RoIs."""

        def is_valid(entry):
            # Valid images have:
            #   (1) At least one foreground RoI OR
            #   (2) At least one background RoI
            overlaps = entry['max_overlaps']
            # find boxes with sufficient overlap
            fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                               (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
            # image is only valid if such boxes exist
            valid = len(fg_inds) > 0 or len(bg_inds) > 0
            return valid

        num = len(roidb)

        filtered_roidb = [entry for entry in roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                           num, num_after)
        return filtered_roidb

    def load_roidb(self):
        if socket.gethostname()=='hexindong':
            cache_file = os.path.join(self._data_path, 'train_gt_roidb_LOCAL.pkl')
            val_cache_file = os.path.join(self._data_path, 'valid_gt_roidb_LOCAL.pkl')
        else:
            cache_file = os.path.join(self._data_path, 'train_gt_roidb_SERVER.pkl')
            val_cache_file = os.path.join(self._data_path, 'valid_gt_roidb_SERVER.pkl')
        if os.path.exists(cache_file) & os.path.exists(val_cache_file):
            print 'Dataset will be loaded from existing cache file'
            with open(cache_file, 'rb') as fid:
                train_roidb = cPickle.load(fid)
            print '  train gt roidb loaded from {}'.format(cache_file)
            with open(val_cache_file, 'rb') as fid:
                valid_roidb = cPickle.load(fid)
            print '  valid gt roidb loaded from {}'.format(val_cache_file)

            return train_roidb, valid_roidb

        print 'Prepare the kitti dataset(train:{}/valid:{}) for training, please wait ...'.format(self.input_num, self.input_val_num)
        train_roidb = [self.load_kitti_annotation(index) for index in self.inputIndex['train_index']]
        valid_roidb = [self.load_kitti_annotation(index) for index in self.inputIndex['valid_index']]

        # 'Func prepare_roidb()':this function add more elements to parameters so the changes will be save to parameters
        # https://www.cnblogs.com/cmnz/p/6927260.html
        train_roidb, valid_roidb = self.prepare_roidb(train_roidb, valid_roidb)

        train_roidb = self.filter_roidb(train_roidb)
        valid_roidb = self.filter_roidb(valid_roidb)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(train_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        with open(val_cache_file, 'wb') as fid:
            cPickle.dump(valid_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(val_cache_file)

        return train_roidb, valid_roidb

    def lidar3d_path_at(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # set the prefix
        prefix = 'velodyne'
        #  lidar3d_path = '$Faster-RCNN_TF/data/KITTI/object/training/velodyne/000000.bin'
        lidar3d_path = os.path.join(self._data_path, prefix, str(index).zfill(6) + '.bin')
        assert os.path.exists(lidar3d_path), 'Path does not exist: {}'.format(lidar3d_path)
        return lidar3d_path

    def lidar_bv_path_at(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        index = int(index)
        # set the prefix
        prefix = 'lidar_bv'
        # lidar_bv_path = '$Faster-RCNN_TF/data/KITTI/object/training/lidar_bv/000000.npy'
        lidar_bv_path = os.path.join(self._data_path, prefix, str(index).zfill(6) + '.npy')
        assert os.path.exists(lidar_bv_path), \
            'Path does not exist: {}'.format(lidar_bv_path)
        return lidar_bv_path

    def calib_at(self, index):
        """
        Return the calib sequence.
        """

        calib_ori = self.load_kitti_calib(index)
        calib = np.zeros((4, 12))
        calib[0, :] = calib_ori['P2'].reshape(12)
        calib[1, :] = calib_ori['P3'].reshape(12)
        calib[2, :9] = calib_ori['R0'].reshape(9)
        calib[3, :] = calib_ori['Tr_velo2cam'].reshape(12)

        return calib

    def get_fileIndex(self, data_path):
        assign_path = osp.join(self._data_path, 'image_assign')
        if not os.path.exists(assign_path):  #
            input_num = len(os.listdir(osp.join(data_path, 'velodyne')))
            test_index = []
            train_index = []
            temp_index = sorted(random.sample(range(input_num), int(input_num * 0.9990)))
            # generate test index
            for i in range(input_num):
                if i not in temp_index:
                    test_index.append(i)
            # generate valid index
            valid_index = sorted(random.sample(temp_index, int(input_num * 0.160)))
            # generate train index
            for k in temp_index:
                if k not in valid_index:
                    train_index.append(k)

            os.mkdir(assign_path)
            file = open(assign_path + '/train.txt', 'w')
            for j in range(len(train_index)):
                file.write(str(train_index[j]))
                file.write('\n')
            file.close()

            file = open(assign_path + '/test.txt', 'w')
            for j in range(len(test_index)):
                file.write(str(test_index[j]))
                file.write('\n')
            file.close()

            file = open(assign_path + '/valid.txt', 'w')
            for j in range(len(valid_index)):
                file.write(str(valid_index[j]))
                file.write('\n')
            file.close()
        else:
            train_index = np.loadtxt(assign_path + '/train.txt', dtype=int)
            # test_index = np.loadtxt(assign_path + '/test.txt', dtype=int)
            valid_index = np.loadtxt(assign_path + '/valid.txt', dtype=int)

        return dict({'train_index': train_index, 'valid_index': valid_index})

    def load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from txt file in the KITTI
        format.
        """
        filename = os.path.join(self._data_path, 'label_2', str(index).zfill(6) + '.txt')
        calib = self.load_kitti_calib(index)  # calib
        Tr = calib['Tr_velo2cam']
        # print 'Loading: {}'.format(filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
        num_objs = len(lines)
        translation = np.zeros((num_objs, 3), dtype=np.float32)
        rys = np.zeros((num_objs), dtype=np.float32)
        lwh = np.zeros((num_objs, 3), dtype=np.float32)
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        boxes_bv = np.zeros((num_objs, 4), dtype=np.float32)
        boxes3D = np.zeros((num_objs, 6), dtype=np.float32)
        boxes3D_lidar = np.zeros((num_objs, 6), dtype=np.float32)
        boxes3D_cam_cnr = np.zeros((num_objs, 24), dtype=np.float32)
        boxes3D_corners = np.zeros((num_objs, 24), dtype=np.float32)
        alphas = np.zeros((num_objs), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, len(self._classes)), dtype=np.float32)
        # Load object bounding boxes into a data frame.
        ix = -1
        for line in lines:
            obj = line.strip().split(' ')
            try:
                cls = self._class_to_ind[obj[0].strip()]
            except:
                continue
            # ignore objects with undetermined difficult level
            # level = self.get_obj_level(obj)
            # if level > 3:
            #     continue
            ix += 1
            # 0-based coordinates
            alpha = float(obj[3])
            x1 = float(obj[4])
            y1 = float(obj[5])
            x2 = float(obj[6])
            y2 = float(obj[7])
            h = float(obj[8])
            w = float(obj[9])
            l = float(obj[10])
            tx = float(obj[11])
            ty = float(obj[12])
            tz = float(obj[13])
            ry = float(obj[14])

            rys[ix] = ry
            lwh[ix, :] = [l, w, h]
            alphas[ix] = alpha
            translation[ix, :] = [tx, ty, tz]
            boxes[ix, :] = [x1, y1, x2, y2]
            boxes3D[ix, :] = [tx, ty, tz, l, w, h]
            # convert boxes3D cam to 8 corners(cam)
            boxes3D_cam_cnr_single = computeCorners3D(boxes3D[ix, :], ry)
            boxes3D_cam_cnr[ix, :] = boxes3D_cam_cnr_single.reshape(24)
            # convert 8 corners(cam) to 8 corners(lidar)
            boxes3D_corners[ix, :] = camera_to_lidar_cnr(boxes3D_cam_cnr_single, Tr)
            # convert 8 corners(lidar) to  lidar boxes3D
            boxes3D_lidar[ix, :] = lidar_cnr_to_3d(boxes3D_corners[ix, :], lwh[ix, :])
            # convert 8 corners(lidar) to lidar bird view
            # avb = lidar_3d_to_bv(boxes3D_lidar[ix, :])
            boxes_bv[ix, :] = my_conner2bvbox(boxes3D_corners[ix, :])  # boxes3D_corners.shape = [3,8]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

            if not self.check_box_bounding(boxes_bv[ix, :]):
                rys[ix] = 0
                lwh[ix, :] = [0, 0, 0]
                alphas[ix] = 0
                translation[ix, :] = [0, 0, 0]
                boxes[ix, :] = [0, 0, 0, 0]
                boxes3D[ix, :] = [0, 0, 0, 0, 0, 0]
                boxes3D_cam_cnr[ix, :] = np.zeros((24), dtype=np.float32)
                boxes3D_corners[ix, :] = np.zeros((24), dtype=np.float32)
                boxes3D_lidar[ix, :] = [0, 0, 0, 0, 0, 0]
                boxes_bv[ix, :] = [0, 0, 0, 0]
                gt_classes[ix] = 0
                overlaps[ix, cls] = 0
                ix = ix - 1

        # rys.resize(ix + 1)
        # lwh.resize(ix + 1, 3)
        # translation.resize(ix + 1, 3)
        # alphas.resize(ix + 1)
        # boxes.resize(ix + 1, 4)
        # boxes_bv.resize(ix + 1, 4)
        # boxes3D.resize(ix + 1, 6)
        # boxes3D_lidar.resize(ix + 1, 6)
        # boxes3D_cam_cnr.resize(ix + 1, 24)
        # boxes3D_corners.resize(ix + 1, 24)
        # gt_classes.resize(ix + 1)
        # overlaps.resize(ix + 1, self.num_classes)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        # if index == '000142':
        #     print(overlaps)

        return {'ry': rys,
                'lwh': lwh,
                'boxes': boxes,
                'boxes_bv': boxes_bv,
                'boxes_3D_cam': boxes3D,
                'boxes_3D': boxes3D_lidar,
                'boxes3D_cam_corners': boxes3D_cam_cnr,
                'boxes_corners': boxes3D_corners,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'xyz': translation,
                'alphas': alphas,
                'flipped': False}

    def load_kitti_calib(self, index):
        """
        load projection matrix
        """
        calib_dir = os.path.join(self._data_path, 'calib', str(index).zfill(6) + '.txt')
        #         P0 = np.zeros(12, dtype=np.float32)
        #         P1 = np.zeros(12, dtype=np.float32)
        #         P2 = np.zeros(12, dtype=np.float32)
        #         P3 = np.zeros(12, dtype=np.float32)
        #         R0 = np.zeros(9, dtype=np.float32)
        #         Tr_velo_to_cam = np.zeros(12, dtype=np.float32)
        #         Tr_imu_to_velo = np.zeros(12, dtype=np.float32)

        #         j = 0
        with open(calib_dir) as fi:
            lines = fi.readlines()
        # assert(len(lines) == 8)

        # obj = lines[0].strip().split(' ')[1:]
        # P0 = np.array(obj, dtype=np.float32)
        # obj = lines[1].strip().split(' ')[1:]
        # P1 = np.array(obj, dtype=np.float32)
        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        # obj = lines[6].strip().split(' ')[1:]
        # P0 = np.array(obj, dtype=np.float32)

        return {'P2': P2.reshape(3, 4),
                'P3': P3.reshape(3, 4),
                'R0': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

    def get_obj_level(self, obj):
        height = float(obj[7]) - float(obj[5]) + 1
        trucation = float(obj[1])
        occlusion = float(obj[2])
        if height >= 40 and trucation <= 0.15 and occlusion <= 0:
            return 1
        elif height >= 25 and trucation <= 0.3 and occlusion <= 1:
            return 2
        elif height >= 25 and trucation <= 0.5 and occlusion <= 2:
            return 3
        else:
            return 4

    def augmentation_of_data(self):
        # Rotation of the image or change the scale
        pass

    def get_minibatch(self, idx=0, name='train'):
        """Given a roidb, construct a minibatch sampled from it."""
        dataset = self.training_rois if name == 'train' else self.validing_rois

        im_scales = [1]
        lidar_bv = np.load(dataset[idx]['lidar_bv_path'])
        lidar_bv_blob = lidar_bv.reshape((1, lidar_bv.shape[0], lidar_bv.shape[1], lidar_bv.shape[2]))
        lidar3d = np.fromfile(dataset[idx]['lidar3d_path'], dtype=np.float32)
        lidar3d_blob = lidar3d.reshape((-1,4))

        gt_inds = np.where(dataset[idx]['gt_classes'] != 0)[0]
        gt_boxes_bv = np.empty((len(gt_inds), 6), dtype=np.float32)
        gt_boxes_bv[:, 0:4] = dataset[idx]['boxes_bv'][gt_inds, :]
        gt_boxes_bv[:, 4] = dataset[idx]['gt_classes'][gt_inds]
        gt_boxes_bv[:, 5] = dataset[idx]['ry'][gt_inds]

        # gt boxes 3d: (x, y, z, l, w, h, cls)
        gt_boxes_3d = np.empty((len(gt_inds), 8), dtype=np.float32)
        gt_boxes_3d[:, 0:6] = dataset[idx]['boxes_3D'][gt_inds, :]
        gt_boxes_3d[:, 6] = dataset[idx]['gt_classes'][gt_inds]

        thetas = dataset[idx]['ry'][gt_inds]
        out_1 = np.where(thetas < -np.pi/2)[0]
        thetas[out_1]+=np.pi
        out_2 = np.where(thetas > np.pi/2)[0]
        thetas[out_2]-=np.pi
        gt_boxes_3d[:, 7]= thetas#TODO:Check
        # if thetas<(-1.5708)
        #     thetas+=3.1415926
        # elif thetas>1.5708
        #     thetas = 3.1415926-thetas
        # gt_boxes_3d[:, 7]=thetas

        # gt boxes corners: (x0, ... x7, y0, y1, ... y7, z0, ... z7, cls)
        gt_boxes_corners = np.empty((len(gt_inds), 25), dtype=np.float32)
        gt_boxes_corners[:, 0:24] = dataset[idx]['boxes_corners'][gt_inds, :]
        gt_boxes_corners[:, 24] = dataset[idx]['gt_classes'][gt_inds]

        blobs = dict({'lidar_bv_data': lidar_bv_blob,
                      'lidar3d_data': lidar3d_blob,
                      'calib': dataset[idx]['calib'],
                      'gt_boxes_bv': gt_boxes_bv,
                      'gt_boxes_3d': gt_boxes_3d,
                      'gt_boxes_corners': gt_boxes_corners,
                      'im_info': np.array([[lidar_bv_blob.shape[1], lidar_bv_blob.shape[2], im_scales[0]]],dtype=np.float32),
                      'serial_num': dataset[idx]['lidar_bv_path'].split('/')[-1][0:6]
                      })

        return blobs

    def check_box_bounding(self, bbox):
        stand = self.bounding_box
        if (bbox[0] > (stand[0] + 1)) & (bbox[1] > (stand[1] + 1)) & (bbox[2] < (stand[2] - 1)) & (
            bbox[3] < (stand[3] - 1)):
            return True
        else:
            return False


class dataset_HANGZHOU_test(object):
    def __init__(self,data_path):
        self.data_path=data_path
        self.calib_dir=path_add(data_path,'Calib')
        self.lidar_dir=path_add(data_path,'HDL64')
        self.lidar_bv_dir=path_add(data_path,'lidar_bv')

        self.lidar_file_list=sorted(os.listdir(self.lidar_dir))
        self.input_num = len(self.lidar_file_list)
        print(blue('DataSet Hangzhou[cnt:{}] load successfully'.format(self.input_num)))

    def get_minibatch(self, indice):

        pc_velo=self._get_lidar_by_name(self.lidar_file_list[indice])
        pc_velo_kitti_like=self.like_kitti_pts(pc_velo,intensity_normalized=True,rota_as_kitti=True)
        bv_file=os.path.join(self.lidar_bv_dir,self.lidar_file_list[indice][:-4]+'.npy')
        pc_bv_image=np.load(bv_file)
        # pc_bv_image_online = point_cloud_2_top(pc_velo_kitti_like)
        img = cv2.imread('/home/likewise-open/SENSETIME/hexindong/ProjectDL/cubic-local/DATASET/Hangzhou/2018-07-06-16-21-48/image_101/2018-07-06-16-23-08-516.png')
        blobs = dict({'lidar_bv_data': pc_bv_image.reshape(-1,pc_bv_image.shape[0],pc_bv_image.shape[1],pc_bv_image.shape[2]),
                      'lidar3d_data': pc_velo_kitti_like,
                      # 'calib': dataset[idx]['calib'],
                      'im_info': np.array([[pc_bv_image.shape[0], pc_bv_image.shape[1], 1]],
                                          dtype=np.float32),
                      'image_data': img
                      })

        return blobs

    def _get_lidar_by_name(self,name):
        lidar_name=path_add(self.lidar_dir,name)
        pc = pcd_loader.from_path(lidar_name).pc_data
        pc.dtype=np.float32

        return pc.reshape(-1,4)

    def rotz(self,points, t):
        ''' Rotation about the z-axis. '''
        if len(points.shape) == 2:
            assert points.shape[-1] == 4, 'Input points shape is not [-1,4]'
        else:
            assert False, 'Input points shape is not [-1,4]'
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[c, -s, 0, 0],
                      [s, c, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1.0]],dtype=np.float32)

        return np.dot(R, points.T).T

    def like_kitti_pts(self,pc,intensity_normalized=False,rota_as_kitti=False,rot_back=False):
        if intensity_normalized:
            pc=pc/np.array([1.,1.,1.,255.],dtype=np.float32)
        if rota_as_kitti:
            if rot_back:
                pc=self.rotz(pc,np.pi/2.)
            else:
                pc=self.rotz(pc,-np.pi/2.)

        return pc


class dataset_STI_train(object):  # read txt files one by one
    def __init__(self, arguments):
        self._type = arguments.imdb_type  # kitti or sti
        self.data_path = '/home/hexindong/DATASET/stidataset/'
        self.folder_list = ['170818-1743-LM120', '170825-1708-LM120', '170829-1743-LM120', '170829-1744-LM120',
                            '1180254121101']
        self._classes = ['unknown', 'smallMot', 'bigMot', 'nonMot', 'pedestrian']
        self.type_to_keep = ['smallMot']
        self.num_classes = len(self._classes)
        self.class_convert = dict(zip(self._classes, xrange(self.num_classes)))
        self.total_roidb = []
        self.filter_roidb = []
        self.percent_train = 0.66
        self.percent_valid = 0.26
        self.train_set, self.valid_set, self.test_set = self.load_dataset()
        self.validing_rois_length=len(self.valid_set)
        self.training_rois_length=len(self.train_set)
        print 'Dataset initialization has been done successfully.'
        time.sleep(3)

    def load_dataset(self):
        Instruction_cache_file = path_add(self.data_path, 'Instruction_cache_data.pkl')
        train_cache_file = path_add(self.data_path, 'train_cache_data.pkl')
        valid_cache_file = path_add(self.data_path, 'valid_cache_data.pkl')
        test_cache_file = path_add(self.data_path, 'test_cache_data.pkl')
        if os.path.exists(train_cache_file) & os.path.exists(valid_cache_file) & os.path.exists(test_cache_file)& os.path.exists(Instruction_cache_file):
            print 'Loaded the STi dataset from pkl cache files ...'

            with open(train_cache_file, 'rb') as fid:
                train_set = cPickle.load(fid)
                print '  train gt set(cnt:{}) loaded from {}'.format(len(train_set),train_cache_file)

            with open(valid_cache_file, 'rb') as fid:
                valid_set = cPickle.load(fid)
                print '  valid gt set(cnt:{}) loaded from {}'.format(len(valid_set),valid_cache_file)

            with open(test_cache_file, 'rb') as fid:
                test_set = cPickle.load(fid)
                print '  test gt set(cnt:{}) loaded from {}'.format(len(test_set),test_cache_file)
            with open(Instruction_cache_file, 'rb') as fid:
                key_points = cPickle.load(fid)
                print 'Notice: the groundtruth range is [{}] meters, the label to keep is {} ,please verify that meets requirement !'\
                      .format(key_points[0],key_points[1],)

            return train_set, valid_set, test_set

        print 'Prepare the STi dataset for training, please wait ...'
        self.total_roidb = self.load_sti_annotation()
        self.filter_roidb = self.filter(self.total_roidb, self.type_to_keep)
        train_set, valid_set, test_set = self.assign_dataset(self.filter_roidb)  # train,valid percent
        with open(Instruction_cache_file, 'wb') as fid:
            cPickle.dump([cfg.DETECTION_RANGE,self.type_to_keep], fid, cPickle.HIGHEST_PROTOCOL)
        with open(train_cache_file, 'wb') as fid:
            cPickle.dump(train_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded train gt roidb(cnt:{}) to {}'.format(len(train_set),train_cache_file)
        with open(valid_cache_file, 'wb') as fid:
            cPickle.dump(valid_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded valid gt roidb(cnt:{}) to {}'.format(len(valid_set),valid_cache_file)
        with open(test_cache_file, 'wb') as fid:
            cPickle.dump(test_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded test gt roidb(cnt:{}) to {}'.format(len(test_set),test_cache_file)

        return train_set, valid_set, test_set

    def load_sti_annotation(self):
        """
        Load points and bounding boxes info from txt file in the KITTI
        format.
        """
        for index, folder in enumerate(self.folder_list):
            libel_fname = path_add(self.data_path, folder, 'label', 'result.txt')
            label = []
            files_names = []
            with open(libel_fname, 'r') as f:
                lines = f.readlines()
            for line in lines:
                files_names.append(self.get_fname_from_label(line))
                line = line.replace('unknown', '0.0').replace('smallMot', '1.0').replace('bigMot', '2.0').replace(
                    'nonMot', '3.0').replace('pedestrian', '4.0')
                object_str = line.translate(None, '\"').split('position:{')[1:]
                label_in_frame = []
                for obj in object_str:
                    f_str_num = re.findall('[-+]?\d+\.\d+', obj)
                    for j, num in enumerate(f_str_num):
                        pass
                        f_str_num[j] = float(num)
                    if j == 10:  # filter the  wrong type label like   type: position
                        label_in_frame.append(f_str_num)
                selected_label = np.array(label_in_frame, dtype=np.float32)
                label.append(
                    selected_label[:, (0, 1, 2, 6, 7, 8, 3, 9)])  # extract the valuable data:x,y,z,l,w,h,theta,type
            if index == 0:
                total_labels = label
                total_fnames = files_names
            else:
                total_labels.extend(label)
                total_fnames.extend(files_names)

        dataset = [dict({'files_list': total_fnames[i], 'labels': total_labels[i]}) for i in range(len(total_fnames))]
        return dataset

    def assign_dataset(self, data):
        cnt = len(data)
        test_index = []
        train_index = []

        temp_index = sorted(random.sample(range(cnt), int(cnt * (self.percent_train + self.percent_valid))))
        for i in range(cnt):
            if i not in temp_index:
                test_index.append(i)
        valid_index = sorted(random.sample(temp_index, int(cnt * self.percent_valid)))
        for k in temp_index:
            if k not in valid_index:
                train_index.append(k)

        train_roidb = [data[k] for k in train_index]
        valid_roidb = [data[k] for k in valid_index]
        test_roidb = [data[k] for k in test_index]

        return train_roidb, valid_roidb, test_roidb

    def filter(self, data, filter_type):
        """Remove roidb entries that out of bounds and category."""

        # numpy:->   x,y,z,l,w,h,theta,type
        def is_valid(dataset):
            boxes = dataset['labels']

            bool_stack = [False for _ in range(boxes.shape[0])]
            for i in range(len(filter_type)):
                res = boxes[:, 7] == float(self.class_convert[filter_type[i]])
                bool_stack = np.logical_or(bool_stack, res)
            bounding =cfg.DETECTION_RANGE
            indice_inside = np.where((boxes[:, 0] >= -bounding) & (boxes[:, 0] <= bounding)
                                     & (boxes[:, 1] >= -bounding) & (boxes[:, 1] <= bounding)
                                     & bool_stack
                                     )[0]
            if len(indice_inside) == 0:
                return False, None
            else:
                return True, boxes[indice_inside]

        keep_indice = []
        num = len(data)
        for index in range(num):
            keep, result = is_valid(data[index])
            if keep:
                data[index]['labels'] = result
                keep_indice.append(index)

        filter_data = [data[k] for k in keep_indice]

        num_after = len(filter_data)
        print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after, num, num_after)
        return filter_data

    def augmentation_of_data(self):
        # Rotation of the image or change the scale
        pass

    def get_minibatch(self, idx=0, name='train'):
        """Given a roidb, construct a minibatch sampled from it."""
        if name == 'train':
            dataset = self.train_set
        elif name == 'valid':
            dataset = self.valid_set
        else:
            dataset = self.test_set

        fname = dataset[idx]['files_list']
        lidar_data = pcd2npScan.from_path(path_add(self.data_path, fname.split('/')[0], 'pcd', fname.split('/')[1]))
        gt_label = dataset[idx]['labels']
        blobs = dict({'lidar3d_data': lidar_data.pc_data,
                      'gt_boxes_3d': gt_label,
                      })
        return blobs

    @staticmethod
    def get_fname_from_label(strings):
        regulars = ['files/\d+-\d+-LM\d+/\d+-\d+-LM\d+_\d+\.pcd', 'files/\d+/\d+_\d+\.pcd']
        for i in range(len(regulars)):
            res = re.findall(regulars[i], strings)
            if len(res) != 0:
                if len(res) == 1:
                    return res[0][6:]
                else:
                    print'File: dataset_sti,function:get_fname_from_label \n  regular expression get more than one qualified file name'
                    exit(23)

class dataset_STI_test(object):  # read txt files one by one
    def __init__(self, arguments):
        self._type = arguments.imdb_type  # kitti or sti
        self.data_path = '/home/hexindong/DATASET/stidataset/'
        self.folder_list = ['170829-1744-LM120', '1180254121101']
        self._classes = ['unknown', 'smallMot', 'bigMot', 'nonMot', 'pedestrian']
        self.type_to_keep = ['unknown', 'smallMot']
        self.num_classes = len(self._classes)
        self.class_convert = dict(zip(self._classes, xrange(self.num_classes)))
        self.total_roidb = []
        self.filter_roidb = []
        self.percent_train = 0.66
        self.percent_valid = 0.26
        self.test_set = self.load_dataset()
        self.testing_rois_length=len(self.test_set)
        print 'Done!'

    def load_dataset(self):
        train_cache_file = path_add(self.data_path, 'train_cache_data.pkl')
        valid_cache_file = path_add(self.data_path, 'valid_cache_data.pkl')
        test_cache_file = path_add(self.data_path, 'test_cache_data.pkl')
        if os.path.exists(train_cache_file) & os.path.exists(valid_cache_file) & os.path.exists(test_cache_file):
            print 'Loaded the STi dataset from pkl cache files ...'
            # with open(train_cache_file, 'rb') as fid:
            #     train_set = cPickle.load(fid)
            #     print '  Train gt set loaded from {}'.format(train_cache_file)
            #
            # with open(valid_cache_file, 'rb') as fid:
            #     valid_set = cPickle.load(fid)
            #     print '  valid gt set loaded from {}'.format(valid_cache_file)

            with open(test_cache_file, 'rb') as fid:
                test_set = cPickle.load(fid)
                print '  test gt set loaded from {}'.format(test_cache_file)

            return test_set

        print 'Prepare the STi dataset for training, please wait ...'
        self.total_roidb = self.load_sti_annotation()
        self.filter_roidb = self.filter(self.total_roidb, self.type_to_keep)
        train_set, valid_set, test_set = self.assign_dataset(self.filter_roidb)  # train,valid percent

        with open(train_cache_file, 'wb') as fid:
            cPickle.dump(train_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded train gt roidb to {}'.format(train_cache_file)
        with open(valid_cache_file, 'wb') as fid:
            cPickle.dump(valid_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded valid gt roidb to {}'.format(valid_cache_file)
        with open(test_cache_file, 'wb') as fid:
            cPickle.dump(test_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded test gt roidb to {}'.format(test_cache_file)

        return test_set

    def load_sti_annotation(self):
        """
        Load points and bounding boxes info from txt file in the KITTI
        format.
        """
        for index, folder in enumerate(self.folder_list):
            libel_fname = path_add(self.data_path, folder, 'label', 'result.txt')
            label = []
            files_names = []
            with open(libel_fname, 'r') as f:
                lines = f.readlines()
            for line in lines:
                files_names.append(self.get_fname_from_label(line))
                line = line.replace('unknown', '0.0').replace('smallMot', '1.0').replace('bigMot', '2.0').replace(
                    'nonMot', '3.0').replace('pedestrian', '4.0')
                object_str = line.translate(None, '\"').split('position:{')[1:]
                label_in_frame = []
                for obj in object_str:
                    f_str_num = re.findall('[-+]?\d+\.\d+', obj)
                    for j, num in enumerate(f_str_num):
                        pass
                        f_str_num[j] = float(num)
                    if j == 10:  # filter the  wrong type label like   type: position
                        label_in_frame.append(f_str_num)
                selected_label = np.array(label_in_frame, dtype=np.float32)
                label.append(
                    selected_label[:, (0, 1, 2, 6, 7, 8, 3, 9)])  # extract the valuable data:x,y,z,l,w,h,theta,type
            if index == 0:
                total_labels = label
                total_fnames = files_names
            else:
                total_labels.extend(label)
                total_fnames.extend(files_names)

        dataset = [dict({'files_list': total_fnames[i], 'labels': total_labels[i]}) for i in range(len(total_fnames))]
        return dataset

    def assign_dataset(self, data):
        cnt = len(data)
        test_index = []
        train_index = []

        temp_index = sorted(random.sample(range(cnt), int(cnt * (self.percent_train + self.percent_valid))))
        for i in range(cnt):
            if i not in temp_index:
                test_index.append(i)
        valid_index = sorted(random.sample(temp_index, int(cnt * self.percent_valid)))
        for k in temp_index:
            if k not in valid_index:
                train_index.append(k)

        train_roidb = [data[k] for k in train_index]
        valid_roidb = [data[k] for k in valid_index]
        test_roidb = [data[k] for k in test_index]

        return train_roidb, valid_roidb, test_roidb

    def filter(self, data, filter_type):
        """Remove roidb entries that out of bounds and category."""

        # numpy:->   x,y,z,l,w,h,theta,type
        def is_valid(dataset):
            boxes = dataset['labels']

            bool_stack = [False for _ in range(boxes.shape[0])]
            for i in range(len(filter_type)):
                res = boxes[:, 7] == float(self.class_convert[filter_type[i]])
                bool_stack = np.logical_or(bool_stack, res)

            indice_inside = np.where((boxes[:, 0] >= -45.) & (boxes[:, 0] <= 45.)
                                     & (boxes[:, 1] >= -45.) & (boxes[:, 1] <= 45.)
                                     & bool_stack
                                     )[0]
            if len(indice_inside) == 0:
                return False, None
            else:
                return True, boxes[indice_inside]

        keep_indice = []
        num = len(data)
        for index in range(num):
            keep, result = is_valid(data[index])
            if keep:
                data[index]['labels'] = result
                keep_indice.append(index)

        filter_data = [data[k] for k in keep_indice]

        num_after = len(filter_data)
        print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after, num, num_after)
        return filter_data

    def augmentation_of_data(self):
        # Rotation of the image or change the scale
        pass

    def get_minibatch(self,idx=0,name='test'):
        """Given a roidb, construct a minibatch sampled from it."""
        dataset = self.test_set
        fname = dataset[idx]['files_list']
        lidar_data = pcd2npScan.from_path(path_add(self.data_path, fname.split('/')[0], 'pcd', fname.split('/')[1]))
        # gt_label = dataset[idx]['labels']
        blobs = dict({'lidar3d_data': lidar_data.pc_data,
                      # 'gt_boxes_3d': gt_label,
                      })
        return blobs

    @staticmethod
    def get_fname_from_label(strings):
        regulars = ['files/\d+-\d+-LM\d+/\d+-\d+-LM\d+_\d+\.pcd', 'files/\d+/\d+_\d+\.pcd']
        for i in range(len(regulars)):
            res = re.findall(regulars[i], strings)
            if len(res) != 0:
                if len(res) == 1:
                    return res[0][6:]
                else:
                    print'File: dataset_sti,function:get_fname_from_label \n  regular expression get more than one qualified file name'
                    exit(23)


def get_data(arguments):
    """Get an imdb (image database) by name."""
    if arguments.method == 'train':
        if arguments.imdb_type == 'kitti':
            return dataset_KITTI_train(arguments)
        elif arguments.imdb_type == 'sti':
            return dataset_STI_train(arguments)
        else:
            return dataset_HANGZHOU_train(cfg.DATA_HANGZHOU_DIR)
    else:
        if arguments.imdb_type == 'kitti':
            return dataset_KITTI_test(arguments)
        elif arguments.imdb_type == 'sti':
            return dataset_STI_test(arguments)
        else:
            return dataset_HANGZHOU_test(cfg.DATA_HANGZHOU_DIR)


if __name__ == '__main__':
    import rospy
    from sensor_msgs.msg import PointCloud
    from visualization_msgs.msg import MarkerArray,Marker
    from tools.data_visualize import Boxes_labels_Gen,PointCloud_Gen,pcd_vispy
    # arg = edict()
    # arg.method = 'train'
    # arg.imdb_type = 'kitti'
    # dataset = get_data(arg)
    # a = dataset.get_minibatch(0,name='train')

    rospy.init_node('rostensorflow')
    pub = rospy.Publisher('prediction', PointCloud, queue_size=1000)
    box_pub = rospy.Publisher('label_boxes', MarkerArray, queue_size=1000)
    rospy.loginfo("ROS begins ...")

    # idx = 0
    # while True:
    #     print 'display frame:{}'.format(idx)
    #     scans = dataset.get_minibatch(idx, name='train')
    #     pointcloud = PointCloud_Gen(scans['lidar3d_data'])
    #     # label_boxes = Boxes_labels_Gen(scans['gt_boxes_3d'],ns='GroundTruth')
    #     # pcd_vispy(scans=scans['lidar3d_data'],boxes=scans['gt_boxes_3d'])
    #     pub.publish(pointcloud)
    #     # box_pub.publish(label_boxes)
    #     idx += 1
    idx = 0
    while True:
        prefix = '/home/likewise-open/SENSETIME/hexindong/DISK1/DATASET/velodyne_points/data/'
        name = prefix +str(idx).zfill(10)+'.bin'
        scan = np.fromfile(name,dtype=np.float32).reshape(-1,4)

        pointcloud = PointCloud_Gen(scan)
        pub.publish(pointcloud)

        idx+=1