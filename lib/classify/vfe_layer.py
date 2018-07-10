
import numpy as np
from network.config import cfg
from classify.rpn_3dcnn import bounding_filter,rot_sca_pc
from tools.data_visualize import pcd_vispy,pcd_show_now,boxary2dic


DEBUG = False

def vfe_cube_Gen(lidarPoints, rpnBoxes, method='train'):
    # rpnBoxes:(x1,y1,z1),(x2,y2,z2),score,cls_label,yaw
    display_stack=[]
    voxel_list = []
    if DEBUG:
        pass
        display_stack.append(pcd_vispy(lidarPoints, boxes=boxary2dic(rpnBoxes),visible=True,multi_vis=False))

    for box in rpnBoxes:
        rpn_points, min_vertex, ctr_vertex = bounding_filter(lidarPoints, box)
        if rpn_points.shape[0] == 0:  # TODO:claude:check: why box has no points
            continue
        pts = voxel_grid(rpn_points, min_vertex, ctr_vertex, cfg)
        print(pts[2])
        if pts[0].shape[0] > 4:  # TODO:claude:declare:box with less 10 voxel wil be drop
            voxel_list.append(pts)
    # for i in range(voxel_list):

    return voxel_list

def voxel_grid(point_clouds, min_vertex,ctr_vertex,cfg):
    # Input:
    #   (N, 3):only x,y,z
    # Output:
    #   voxel_dict
    np.random.shuffle(point_clouds)

    point_cloud_min = point_clouds - min_vertex
    point_cloud_ctr = point_clouds - ctr_vertex

    voxel_size = np.array(cfg.CUBIC_RES, dtype=np.float32)
    voxel_index = np.floor(point_cloud_min[:,0:3]/voxel_size).astype(np.int)
    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)
    K = len(coordinate_buffer)
    T = cfg.VOXEL_POINT_COUNT

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(K), dtype=np.int64)

    # [K, T, 6] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, T, 6), dtype=np.float32)
    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud_ctr):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number == 0:
            feature_buffer[index, :, :3] = point[:3]
        if number < T:
            feature_buffer[index, number, :3] = point[:3]
            number_buffer[index] += 1

    # ctr = feature_buffer[3, :number_buffer[3], :3].sum(axis=0)/ number_buffer[3]
    for idx in range(K):
        center = feature_buffer[idx, :number_buffer[idx], :3].sum(axis=0) / number_buffer[idx]
        feature_buffer[idx, :, -3:] = feature_buffer[idx, :, :3] - center#TODO:to learn from it

    # voxel_dict = {'feature_buffer': feature_buffer,
    #               'coordinate_buffer': coordinate_buffer,
    #               'number_buffer': number_buffer}
    voxel_dict = [feature_buffer, coordinate_buffer,number_buffer]
    return voxel_dict


if __name__ == '__main__':
    from easydict import EasyDict as edict
    from dataset.dataset import get_data
    arg = edict()
    arg.method = 'train'
    arg.imdb_type = 'kitti'
    dataset = get_data(arg)
    for i in range(10):
        blob = dataset.get_minibatch(i,name='train')
        res = vfe_cube_Gen(blob['lidar3d_data'],blob['gt_boxes_3d'])
        a = 0
