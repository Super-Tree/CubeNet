# coding=utf-8
import os
import numpy as np
import cPickle
from tools.data_visualize import pcd_show_now,pcd_vispy_standard

DEBUG = False
class DataLoader(object):
    def __init__(self,path_,types=('__background__', 'Car', 'Pedestrian', 'Cyclist','Van')):
        self._data_path=path_
        self._classes = types
        self.num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))

    def get_minibatch(self, idx=0):
        """Given a roidb, construct a minibatch sampled from it."""
        label = self.load_kitti_annotation(idx)

        lidar3d = np.fromfile(os.path.join(self._data_path,'velodyne',str(idx).zfill(6)+'.bin'), dtype=np.float32)
        lidar3d_blob = lidar3d.reshape((-1,4))

        # return self.dic2array(label),lidar3d_blob
        return label,lidar3d_blob

    def computeCorners3D(self,Boxex3D, ry):

        # compute rotational matrix around yaw axis
        R = np.array([[np.cos(ry), 0, np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, np.cos(ry)]]).reshape((3, 3))

        # 3D bounding box dimensions
        l, w, h = Boxex3D[3:6]
        x, y, z = Boxex3D[0:3]

        # 3D bounding box corners
        x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
        y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
        z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

        corners = np.vstack((x_corners, y_corners, z_corners))

        # rotate and translate 3D bounding box
        corners_3D = np.dot(R, corners)
        corners_3D[0, :] = corners_3D[0, :] + x
        corners_3D[1, :] = corners_3D[1, :] + y
        corners_3D[2, :] = corners_3D[2, :] + z

        return corners_3D

    def camera_to_lidar_cnr(self,pts_3D, Tr,R0):
        """
        convert camera corners to lidar corners
        """

        def inverse_rigid_trans(Tr):
            ''' Inverse a rigid body transform matrix (3x4 as [R|t])
                [R'|-R't; 0|1]
            '''
            inv_Tr = np.zeros_like(Tr)  # 3x4
            inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
            inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
            return inv_Tr

        if pts_3D.shape[1] == 24:
            pts_3D = pts_3D.reshape((3, 8))

        pts_3D= np.dot(np.linalg.inv(R0),(pts_3D))

        pts_3D = np.vstack((pts_3D, np.ones(8)))

        assert pts_3D.shape == (4, 8)

        # R = np.linalg.inv(P[:, :3])
        # # T = -P[:, 3].reshape((3, 1))
        # T = np.zeros((3, 1))
        # T[0] = -P[1, 3]
        # T[1] = -P[2, 3]
        # T[2] = P[0, 3]
        # RT = np.hstack((R, T))
        RT = inverse_rigid_trans(Tr)
        lidar_corners = np.dot(RT, pts_3D)
        lidar_corners = lidar_corners[:3, :]

        return lidar_corners.reshape(-1, 24)

    def lidar_cnr_to_3d(self,corners, lwh):
        """
        lidar_corners to Boxex3D
        """
        shape = corners.shape
        if shape[0] == 24:
            boxes_3d = np.zeros(6)
            corners = corners.reshape((3, 8))
            boxes_3d[:3] = corners.mean(1)
            boxes_3d[3:] = lwh
        else:
            boxes_3d = np.zeros((shape[0], 6))
            corners = corners.reshape((-1, 3, 8))
            boxes_3d[:, :3] = corners.mean(2)
            boxes_3d[:, 3:] = lwh
        return boxes_3d

    def load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from txt file in the KITTI
        format.
        """
        filename = os.path.join(self._data_path, 'label_2', str(index).zfill(6) + '.txt')
        calib = self.load_kitti_calib(index)  # calib
        Tr = calib['Tr_velo2cam']
        R0 = calib['R0']

        with open(filename, 'r') as f:
            lines = f.readlines()
        num_objs=0
        for line in lines:
            elements=line.split()
            if elements[0] in self._classes:
                num_objs+=1
        # num_objs = len(lines)
        translation = np.zeros((num_objs, 3), dtype=np.float32)
        rys = np.zeros((num_objs), dtype=np.float32)
        lwh = np.zeros((num_objs, 3), dtype=np.float32)
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        boxes3D = np.zeros((num_objs, 6), dtype=np.float32)
        boxes3D_lidar = np.zeros((num_objs, 6), dtype=np.float32)
        boxes3D_cam_cnr = np.zeros((num_objs, 24), dtype=np.float32)
        boxes3D_corners = np.zeros((num_objs, 24), dtype=np.float32)
        alphas = np.zeros((num_objs), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
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
            boxes3D_cam_cnr_single = self.computeCorners3D(boxes3D[ix, :], ry)
            boxes3D_cam_cnr[ix, :] = boxes3D_cam_cnr_single.reshape(24)
            # convert 8 corners(cam) to 8 corners(lidar)
            boxes3D_corners[ix, :] = self.camera_to_lidar_cnr(boxes3D_cam_cnr_single, Tr,R0)
            # convert 8 corners(lidar) to  lidar boxes3D
            boxes3D_lidar[ix, :] = self.lidar_cnr_to_3d(boxes3D_corners[ix, :], lwh[ix, :])
            # convert 8 corners(lidar) to lidar bird view
            # avb = lidar_3d_to_bv(boxes3D_lidar[ix, :])
            gt_classes[ix] = cls

            # if not self.check_box_bounding(boxes_bv[ix, :]):
            #     rys[ix] = 0
            #     lwh[ix, :] = [0, 0, 0]
            #     alphas[ix] = 0
            #     translation[ix, :] = [0, 0, 0]
            #     boxes[ix, :] = [0, 0, 0, 0]
            #     boxes3D[ix, :] = [0, 0, 0, 0, 0, 0]
            #     boxes3D_cam_cnr[ix, :] = np.zeros((24), dtype=np.float32)
            #     boxes3D_corners[ix, :] = np.zeros((24), dtype=np.float32)
            #     boxes3D_lidar[ix, :] = [0, 0, 0, 0, 0, 0]
            #     boxes_bv[ix, :] = [0, 0, 0, 0]
            #     gt_classes[ix] = 0
            #     ix = ix - 1

        return {'ry': rys,
                'lwh': lwh,
                'boxes': boxes,
                # 'boxes_3D_cam': boxes3D,
                'boxes_3D': boxes3D_lidar,
                # 'boxes3D_cam_corners': boxes3D_cam_cnr,
                'boxes_corners': boxes3D_corners,
                'gt_classes': gt_classes,
                'xyz': translation,
                'alphas': alphas,}

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

    def dic2array(self,label):
        # one box: type,xyz,lwh,yaw, score,reserve1,reserve2]
        return np.hstack((label['gt_classes'].reshape(-1, 1), label['boxes_3D'], -1 * (label['ry'].reshape(-1, 1) + np.pi / 2.)))


class BoxFactory(object):
    def __init__(self, arg):
        self.arg = arg
        self.dataset = DataLoader(self.arg.path)
        self.data_cnt=0

        self.save_folder=os.path.join(self.arg.out_path, 'gt_points_positive')
        self.points_file=os.path.join(self.save_folder,'GT_Points_Positive.pkl')
        self.points_label_file=os.path.join(self.save_folder,'GT_Points_Positive_Label.pkl')
        self.points_index_file=os.path.join(self.save_folder,'GT_Points_Positive_Index.pkl')
        self.points_info_file=os.path.join(self.save_folder,'GT_Points_Positive_Info.pkl')

        self.cubes_save_folder=os.path.join(self.arg.out_path, 'gt_cubes_positive')
        self.cubes_file=os.path.join(self.cubes_save_folder,'GT_Cubes_Positive.pkl')
        self.cubes_label_file=os.path.join(self.cubes_save_folder,'GT_Cubes_Positive_Label.pkl')
        self.cubes_info_file=os.path.join(self.cubes_save_folder,'GT_Cubes_Positive_Info.pkl')

        self.cubes_filter_save_folder=os.path.join(self.arg.out_path, 'gt_filtered_cubes')
        self.cubes_filter_file=os.path.join(self.cubes_filter_save_folder,'GT_Filtered_Cubes_Positive.pkl')
        self.cubes_filter_label_file=os.path.join(self.cubes_filter_save_folder,'GT_Cubes_Filtered_Positive_Label.pkl')
        self.cubes_filter_info_file=os.path.join(self.cubes_filter_save_folder,'GT_Filtered_Cubes_Info.pkl')

    def gt_points_generate_run(self):
        data_cnt=len(os.listdir(os.path.join(self.arg.path,'velodyne')))
        cube_points=[]
        cube_points_label=[]
        cube_points_index=[]

        for idx in range(data_cnt):
            label,lidar3d =self.dataset.get_minibatch(idx)
            label_ary=self.dataset.dic2array(label)
            self.conners = label['boxes_corners'].reshape(-1,3,8).transpose(0,2,1).reshape(-1,3)## tmp to show
            points_list,points_label_list=self.boundingbox_filter(lidar3d, label_ary)

            cube_points.extend(points_list)
            cube_points_label.extend(points_label_list)
            cube_points_index.extend((np.ones(len(points_list))*idx).tolist())

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        with open(self.points_file, 'w') as fid:
            cPickle.dump(cube_points,fid,cPickle.HIGHEST_PROTOCOL)
        with open(self.points_label_file, 'w') as fid:
            cPickle.dump(cube_points_label,fid,cPickle.HIGHEST_PROTOCOL)
        with open(self.points_index_file, 'w') as fid:
            cPickle.dump(cube_points_index,fid,cPickle.HIGHEST_PROTOCOL)
        with open(self.points_info_file, 'w') as fid:
            cPickle.dump(self.arg.anchor,fid,cPickle.HIGHEST_PROTOCOL)
        print("Done with generating gt_points")

    def gt_cube_generate_run(self):
        if os.path.exists(self.points_file) and os.path.exists(self.points_label_file) and os.path.exists(self.points_index_file) and os.path.exists(self.points_info_file):
            print "Starting to load existing pkl gt_points ..."
            with open(self.points_file, 'rb') as fid:
                points_file=cPickle.load(fid)
            with open(self.points_label_file, 'rb') as fid:
                points_label_file=cPickle.load(fid)
            with open(self.points_index_file, 'rb') as fid:
                points_index_file=cPickle.load(fid)
            with open(self.points_info_file, 'rb') as fid:
                points_info_file=cPickle.load(fid)
            print "   Data cnt:[{}],with bounding info:{}".format(len(points_file),points_info_file)
        else:
            print "The existing data is not complete!"
            return 0

        cubes=[]
        cubes_label=[]
        data_cnt=len(points_file)
        for idx in range(data_cnt):
            label,lidar3d =points_label_file[idx],points_file[idx]
            cube,cube_label = self.points2cube(lidar3d,self.arg.cube_resolution,label)
            cubes.append(cube)
            cubes_label.append(cube_label)
        cubes_np=np.array(cubes)
        cubes_label_np=np.array(cubes_label)
        if not os.path.exists(self.cubes_save_folder):
            os.makedirs(self.cubes_save_folder)
        with open(self.cubes_file, 'w') as fid:
            cPickle.dump(cubes_np,fid,cPickle.HIGHEST_PROTOCOL)
        with open(self.cubes_label_file, 'w') as fid:
            cPickle.dump(cubes_label_np,fid,cPickle.HIGHEST_PROTOCOL)
        with open(self.cubes_info_file, 'w') as fid:
            tmp=self.arg.anchor
            tmp.extend(self.arg.cube_resolution)
            cPickle.dump(tmp,fid,cPickle.HIGHEST_PROTOCOL)
        print("Done with generating gt_cubes")

    def gt_cube_filter_generate_run(self):
        if os.path.exists(self.cubes_file) and os.path.exists(self.cubes_label_file):
            print "Starting to load existing pkl gt_cubes ..."
            with open(self.cubes_file, 'rb') as fid:
                cubes_np = cPickle.load(fid)
            with open(self.cubes_label_file, 'rb') as fid:
                cubes_label_np = cPickle.load(fid)
            with open(self.cubes_info_file, 'rb') as fid:
                cubes_info = cPickle.load(fid)
            print "   Data cnt:[{}],with cube info:{}".format(len(cubes_np), cubes_info)
        else:
            print "The cube pkl data is not existing!"
            return 0

        index=[]
        cubes_np=cubes_np[0:300]
        for idx,cube in enumerate(cubes_np):
            sum_result=cube.sum()
            if sum_result>self.arg.cube_min_pts and (cubes_label_np[idx,0]==1 or cubes_label_np[idx,0]==4):
                index.append(idx)
        cubes_filtered_np=cubes_np[index]
        cubes_filtered_label_np=cubes_label_np[index]

        if not os.path.exists(self.cubes_filter_save_folder):
            os.makedirs(self.cubes_filter_save_folder)
        with open(self.cubes_filter_file, 'w') as fid:
            cPickle.dump(cubes_filtered_np, fid, cPickle.HIGHEST_PROTOCOL)
        with open(self.cubes_filter_label_file, 'w') as fid:
            cPickle.dump(cubes_filtered_label_np, fid, cPickle.HIGHEST_PROTOCOL)
        with open(self.cubes_filter_info_file, 'w') as fid:
            tmp=self.arg.anchor
            tmp.extend(self.arg.cube_resolution)
            tmp.append(self.arg.cube_min_pts)
            cPickle.dump(tmp, fid, cPickle.HIGHEST_PROTOCOL)
        print("Done with generating filtered gt_cubes")

    def gt_cube_loader(self):
        if os.path.exists(self.cubes_file) and os.path.exists(self.cubes_label_file):
            print "Starting to load existing pkl gt_cubes ..."
            with open(self.cubes_file, 'rb') as fid:
                cubes_np = cPickle.load(fid)
            with open(self.cubes_label_file, 'rb') as fid:
                cubes_label_np = cPickle.load(fid)
            with open(self.cubes_info_file, 'rb') as fid:
                cubes_info = cPickle.load(fid)
            print "   Data cnt:[{}],with cube info:{}".format(len(cubes_np), cubes_info)
        else:
            print "The cube pkl data is not existing!"
            return 0

        return cubes_np

    def gt_filtered_cube_loader(self):
        if os.path.exists(self.cubes_file) and os.path.exists(self.cubes_label_file):
            print "Starting to load existing pkl filtered gt_cubes ..."
            with open(self.cubes_filter_file, 'rb') as fid:
                cubes_filter_np = cPickle.load(fid)
            with open(self.cubes_filter_label_file, 'rb') as fid:
                cubes_filter_label_np = cPickle.load(fid)
            with open(self.cubes_filter_info_file, 'rb') as fid:
                cubes_filter_info = cPickle.load(fid)
            print "   Data cnt:[{}],with cube info:{}".format(len(cubes_filter_np), cubes_filter_info)
        else:
            print "The cube filtered pkl data is not existing!"
            return 0

    def boundingbox_filter(self, lidarPoints, gt_box):

        def bounding_filter(points, box):
            # one box: type,xyz,lwh,yaw, [score,reserve1,reserve2]
            x_min = box[1] - arg.anchor[0] / 2.
            x_max = box[1] + arg.anchor[0] / 2.
            y_min = box[2] - arg.anchor[1] / 2.
            y_max = box[2] + arg.anchor[1] / 2.
            z_min = box[3] - arg.anchor[2] / 2.
            z_max = box[3] + arg.anchor[2] / 2.

            x_points = points[:, 0]
            y_points = points[:, 1]
            z_points = points[:, 2]
            f_filt = np.logical_and((x_points > x_min), (x_points < x_max))
            s_filt = np.logical_and((y_points > y_min), (y_points < y_max))
            z_filt = np.logical_and((z_points > z_min), (z_points < z_max))
            fliter = np.logical_and(np.logical_and(f_filt, s_filt), z_filt)
            indice = np.flatnonzero(fliter)
            filter_points = points[indice]

            return filter_points, np.array([x_min, y_min, z_min, 0.], dtype=np.float32), np.array(
                [box[1], box[2], box[3], 0.], dtype=np.float32)

        # one box: type,xyz,lwh,yaw, [score,reserve1,reserve2]
        cube_points = []
        cube_points_label = []
        display_stack=[]

        for iidx,box in enumerate(gt_box):
            bounding_points,min_vertex,ctr_vertex = bounding_filter(lidarPoints,box)
            bounding_ctr_points=bounding_points - ctr_vertex
            cube_points.append(bounding_ctr_points)
            cube_points_label.append(box)
            if DEBUG:
                display_stack.append(pcd_vispy_standard(bounding_ctr_points,name='points_'+str(iidx), boxes=box-[0,box[1],box[2],box[3],0,0,0,0],visible=False,point_size =0.1,multi_vis=True))
            # break
        if DEBUG:
            display_stack.append(pcd_vispy_standard([lidarPoints,self.conners],name="lidar full scope", boxes=gt_box,visible=False,multi_vis=True))
            pcd_show_now()

        return cube_points,cube_points_label

    def points2cube(self,points_ctr,resolution,label):
        min_vertex = np.array([-self.arg.anchor[0]/2,-self.arg.anchor[1]/2,-self.arg.anchor[2]/2,0])
        points_mv_min = np.subtract(points_ctr, min_vertex)  # using fot coordinate
        cubic_size = [int(np.ceil(np.round(self.arg.anchor[i] / resolution[i], 3))) for i in range(3)]  # Be careful about python number decimal
        cubic_size.append(1)

        x_cub = np.divide(points_mv_min[:, 0], resolution[0]).astype(np.int32)
        y_cub = np.divide(points_mv_min[:, 1], resolution[1]).astype(np.int32)
        z_cub = np.divide(points_mv_min[:, 2], resolution[2]).astype(np.int32)

        feature = np.ones([len(points_mv_min[:, 3]), 1], dtype=np.float32)

        cubic_feature = np.zeros(shape=cubic_size, dtype=np.float32)
        cubic_feature[x_cub, y_cub, z_cub] = feature  # TODO:select&add feature # points_mv_ctr  # using center coordinate system

        if DEBUG:
            # feature = np.hstack((x_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[0]/2),y_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[1]/2),z_cub.reshape(-1,1)-(cfg.CUBIC_SIZE[2]/2),points_mv_ctr_rot[:,3].reshape(-1,1))) #points_mv_ctr_rot
            feature_position = np.hstack((x_cub.reshape(-1, 1) - (cubic_size[0] / 2),
                                 y_cub.reshape(-1, 1) - (cubic_size[1] / 2),
                                 z_cub.reshape(-1, 1) - (cubic_size[2] / 2)))
            print("Points in cube:{}".format(x_cub.shape[0]))
            bounding_box=np.array([5,0,0,0,cubic_size[0],cubic_size[1],cubic_size[2],0])

            gt_box=np.array([label[0],0,0,0,label[4]/resolution[0],label[5]/resolution[0],label[6]/resolution[0],label[7]])
            gt_real_box=np.array([label[0],0,0,0,label[4],label[5],label[6],label[7]])
            box=np.vstack((bounding_box,gt_box,gt_real_box))
            pcd_vispy_standard(scans=[feature_position,points_ctr],boxes=box,point_size=0.1)

        label_cube=label
        # label_cube=gt_box
        return cubic_feature,label_cube


if __name__ == '__main__':

    arg = type('', (), {})()
    arg.imdb_type = 'kitti'
    arg.path = '/home/likewise-open/SENSETIME/hexindong/ProjectDL/cubic-local/DATASET/KITTI/object/training'
    arg.out_path = '/home/likewise-open/SENSETIME/hexindong/DISK1/DATASET/KITTI/object/CUBE'
    arg.cube_resolution = [0.136, 0.136, 0.14]  # 30x30x15  # car size [0.2858,0.2858,0.1429]:14x14x14
    arg.anchor=[4.,4.,2.]  # TODO:declaration:should be same with config.h
    arg.cube_min_pts=40

    processor=BoxFactory(arg)
    # processor.gt_points_generate_run()
    # processor.gt_cube_generate_run()  # it has a vispy viewer
    # processor.gt_cube_filter_generate_run()
    # processor.gt_filtered_cube_loader()
    cnt_array = np.array([x.sum() for x in processor.gt_cube_loader()])

    cnt_array_np=np.array(cnt_array,dtype=np.int32)
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print(cnt_array_np.max())
    # plt.style.use('ggplot')

    cnt_array_np[cnt_array_np>399]=399
    plt.hist(cnt_array_np,
             range=(0,400),
             bins=10,
             color='steelblue',
             edgecolor='k',
             label='PtsNum of Cube')
    plt.tick_params(top='off', right='off')
    plt.legend()
    # plt.savefig('distribution.png')
    plt.show()