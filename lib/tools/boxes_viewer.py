import re
import os
import numpy as np
import vispy
from vispy.scene import visuals
from data_visualize import line_box

def boxes_vispy(boxes=None, name=None,vis_size=(800, 600)):

    canvas = vispy.scene.SceneCanvas(title=name, keys='interactive', size=vis_size,show=True)

    grid = canvas.central_widget.add_grid()
    vb = grid.add_view(row=0, col=0, row_span=2)

    vb.camera = 'turntable'
    vb.camera.elevation = 21.0
    vb.camera.center = (6.5, -0.5, 9.0)
    vb.camera.azimuth = -75.5
    vb.camera.scale_factor = 32.7

    axis = visuals.XYZAxis()
    vb.add(axis)

    if boxes is not None:
        if len(boxes.shape) ==1:
            boxes = boxes.reshape(1,-1)

        det_indice = np.where(boxes[:, -1] == 11)[0]
        det_cnt = len(det_indice)

        gt_cnt = boxes.shape[0]-det_cnt

        for k,box in enumerate(boxes):
            color = (0, 1, 1, 1)  # Green
            if box[-1] == 11:  #  det boxes
                vsp_box = visuals.Box(width=box[3],  depth=box[4],height=box[5], color=(0.3, 0.4, 0.0, 0.06),edge_color='pink')
                mesh_box = vsp_box.mesh.mesh_data
                mesh_border_box = vsp_box.border.mesh_data
                vertices = mesh_box.get_vertices()
                center = np.array([box[0], box[1], box[2]], dtype=np.float32)
                vtcs = np.add(vertices, center)
                mesh_border_box.set_vertices(vtcs)
                mesh_box.set_vertices(vtcs)
                vb.add(vsp_box)
                text = visuals.Text(text='det: ({}/{})'.format(k+1, det_cnt), color='pink', face='OpenSans', font_size=6,
                                        pos=[box[0], box[1], box[2]],anchor_x='left', anchor_y='top', font_manager=None)
                vb.add(text)
            elif box[-1] == 1:# gt smallMot
                 vb.add(line_box(box, color='yellow'))
                 text = visuals.Text(text='gt: ({}/{})'.format(k+1-det_cnt, gt_cnt), color='yellow', face='OpenSans', font_size=8,
                                        pos=[box[0], box[1], box[2]],anchor_x='left', anchor_y='top', font_manager=None)
                 vb.add(text)
            elif box[-1] == 2:# gt smallMot
                 vb.add(line_box(box, color='green'))
                 text = visuals.Text(text='gt: ({}/{})'.format(k+1-det_cnt, gt_cnt), color='green', face='OpenSans', font_size=8,
                                        pos=[box[0], box[1], box[2]],anchor_x='left', anchor_y='top', font_manager=None)
                 vb.add(text)
            elif box[-1] == 3:# gt smallMot
                 vb.add(line_box(box, color='red'))
                 text = visuals.Text(text='gt: ({}/{})'.format(k+1-det_cnt, gt_cnt), color='red', face='OpenSans', font_size=8,
                                        pos=[box[0], box[1], box[2]],anchor_x='left', anchor_y='top', font_manager=None)
                 vb.add(text)
            elif box[-1] == 4:# gt smallMot
                 vb.add(line_box(box, color='blue'))
                 text = visuals.Text(text='gt: ({}/{})'.format(k + 1 - det_cnt, gt_cnt), color='blue', face='OpenSans',
                                font_size=8,
                                pos=[box[0], box[1], box[2]], anchor_x='left', anchor_y='top', font_manager=None)
                 vb.add(text)
            # text = visuals.Text(text=str(k), color=color, face='OpenSans', font_size=12,
            #                     pos=[box[0]-box[3]/2, box[1]-box[4]/2, box[2]-box[5]/2], anchor_x='left', anchor_y='top', font_manager=None)
            # vb.add(text)

    vispy.app.run()

    return canvas

def gt_load(label_fname):
    """
    Load points and bounding boxes info from txt file in the KITTI
    format.
    """
    with open(label_fname, 'r') as f:
        lines = f.readlines()

    label=[]
    for line in lines:
        line = line.replace('unknown', '0.0').replace('smallMot', '1.0').replace('bigMot', '2.0').replace('nonMot', '3.0').replace('pedestrian', '4.0')
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
        tmp = selected_label[:, (0, 1, 2, 6, 7, 8, 3, 9)]
        keep = np.where(tmp[:,-1]!=0)
        tmp = tmp[keep]
        label.append(tmp)  # extract the valuable data:x,y,z,l,w,h,theta,type

    return label

def det_load(det_path):
    files_name = sorted(os.listdir(det_path))
    all_detections=[]

    for idx,f_name in enumerate(files_name):
        boxes = []
        with open(det_path+f_name, 'r') as f:
            lines = f.readlines()
            for one_line in lines:
                data = one_line.split()
                if data[0]!='unknown':
                    boxes.append(np.array([float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),float(data[6]),float(data[7]),11.0]))

        all_detections.append(np.array(boxes))

    return all_detections


if __name__ =='__main__':
    det_path = "/home/hexindong/Desktop/KITTI_Evaluation/kittiEval/data/detections3/"
    label_fname="/home/hexindong/Desktop/KITTI_Evaluation/kittiEval/data/label/result.txt"
    gt = gt_load(label_fname)
    det = det_load(det_path)
    for i in range(10):
        boxes = np.vstack((det[i],gt[i]))
        boxes_vispy(boxes=boxes)