import os
import socket
if socket.gethostname() == "szstdzcp0325":
    import vispy.app
    v = vispy.app.Canvas()
    from vispy.scene import visuals
    import vispy.io as vispy_file

import cv2
from tools.utils import scales_to_255
import tensorflow as tf
import numpy as np
from network.config import cfg

from os.path import join as path_add

if socket.gethostname()=="szstdzcp0325":
    import vispy.app
    v = vispy.app.Canvas()
    vispy.set_log_level('CRITICAL', match='-.-')

#  common functions  ===========================
def BoxAry_Theta(gt_box3d=None,pre_box3d=None,pre_theta_value=None,pre_cube_cls=None):
    # gt_box3d: (x1,y1,z1),(x2,y2,z2),dt_cls,yaw
    # pre_box3d: (x1,y1,z1),(x2,y2,z2),score,rpn_cls_label
    # cubic_theta_value:pre_box3d's yaw value
    boxes=dict({})
    if gt_box3d is None:
        gt_box3d=np.zeros([1,8],dtype=np.float32)
    if pre_box3d is None:
        pre_box3d=np.zeros([cfg.TRAIN.RPN_POST_NMS_TOP_N,8],dtype=np.float32)
    if pre_theta_value is None:
        pre_theta_value=np.ones([cfg.TRAIN.RPN_POST_NMS_TOP_N,1],dtype=np.float32)*(-1.57)
    if pre_cube_cls is None:
        pre_cube_cls = np.zeros([cfg.TRAIN.RPN_POST_NMS_TOP_N, 1], dtype=np.float32)

    boxes["center"]= np.vstack((gt_box3d[:,0:3],pre_box3d[:,0:3]))
    boxes["size"]  = np.vstack((gt_box3d[:,3:6], pre_box3d[:,3:6]))
    boxes["score"]  = np.vstack((gt_box3d[:, 6:7], pre_box3d[:, 6:7]))
    boxes["cls_rpn"]  = np.vstack((gt_box3d[:, 6:7]*4, pre_box3d[:, 7:8]))#two cls flag  to save more information
    boxes["cls_cube"]  = np.vstack((gt_box3d[:, 6:7]*4, np.reshape(pre_cube_cls,[-1,1])*2))#todo add cubic cls
    boxes["yaw"]   = np.vstack((gt_box3d[:, 7:8], np.reshape(pre_theta_value,[-1,1])))#pre_box3d[:, 8:9]

    return boxes
def box3d_2conner(box,rot):
    #box : x,y,z,l,w,h,rot
    vertices = np.zeros([8,3],dtype=np.float32)
    vertices[0] = np.array([0 - float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])
    vertices[1] = np.array([0 - float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])
    vertices[2] = np.array([0 + float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])
    vertices[3] = np.array([0 + float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 - float(box[5]) / 2.0, ])

    vertices[4] = np.array([0 - float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])
    vertices[5] = np.array([0 - float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])
    vertices[6] = np.array([0 + float(box[3]) / 2.0, 0 + float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])
    vertices[7] = np.array([0 + float(box[3]) / 2.0, 0 - float(box[4]) / 2.0, 0 + float(box[5]) / 2.0, ])

    vertices = box_rot_trans(vertices, rot, [box[0],box[1],box[2]])

    return vertices[0],vertices[1],vertices[2],vertices[3],vertices[4],vertices[5],vertices[6],vertices[7],
def boxary2dic(gt_box3d):
    # gt_box3d: (x1,y1,z1),(x2,y2,z2),cls,yaw
    boxes=dict({})
    if len(gt_box3d.shape)==1:
        gt_box3d=gt_box3d.reshape(-1,gt_box3d.shape[0])
    boxes["center"]= gt_box3d[:,0:3]
    boxes["size"]  = gt_box3d[:,3:6]
    boxes["score"]  = gt_box3d[:,6:7]
    boxes["cls_rpn"]  = np.ones([gt_box3d.shape[0],1],dtype=np.float32)
    boxes["cls_cube"]  = np.ones([gt_box3d.shape[0],1],dtype=np.float32)
    boxes["yaw"]   = gt_box3d[:,7:8]

    return boxes
def lidar_3d_to_corners(pts_3D):
    """
    convert pts_3D_lidar (x, y, z, l, w, h) to
    8 corners (x0, ... x7, y0, ...y7, z0, ... z7)
    """

    l = pts_3D[:, 3]
    w = pts_3D[:, 4]
    h = pts_3D[:, 5]

    l = l.reshape(-1, 1)
    w = w.reshape(-1, 1)
    h = h.reshape(-1, 1)

    # clockwise, zero at bottom left
    x_corners = np.hstack((l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.))
    y_corners = np.hstack((w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.))
    z_corners = np.hstack((-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.))

    corners = np.hstack((x_corners, y_corners, z_corners))

    corners[:, 0:8] = corners[:, 0:8] + pts_3D[:, 0].reshape((-1, 1)).repeat(8, axis=1)
    corners[:, 8:16] = corners[:, 8:16] + pts_3D[:, 1].reshape((-1, 1)).repeat(8, axis=1)
    corners[:, 16:24] = corners[:, 16:24] + pts_3D[:, 2].reshape((-1, 1)).repeat(8, axis=1)

    return corners
def video_producer():
    img_path = '/home/hexindong/ws_dl/pyProj/cubic-local/test_result/OFHZ/'

    f = os.listdir(img_path)
    f.sort()
    videoWriter = cv2.VideoWriter('/home/hexindong/Desktop/Drive0064-Man.mp4',cv2.cv.CV_FOURCC(*'PIM1'),fps=20, frameSize=(800, 600))

    for i in f:
        img = cv2.imread(img_path + i)
        cv2.imshow('testResult',img)
        videoWriter.write(img)
        print str(i)

    videoWriter.release()
#  using vispy ============================
class pcd_vispy_client(object):# TODO: qt-client TO BE RE-WRITE
    def __init__(self,QUEUE,scans=None,title=None, keys='interactive', size=(800,600)):
        self.queue=QUEUE
        # self.canvas = vispy.scene.SceneCanvas(title=title, keys=keys, size=size, show=True)
        # self.grid = self.canvas.central_widget.add_grid()
        # self.vb = self.grid.add_view(row=0, col=0, row_span=2)
        # self.vb_img = self.grid.add_view(row=1, col=0)
        #
        # self.vb.camera = 'turntable'
        # self.vb.camera.elevation = 21.0
        # self.vb.camera.center = (6.5, -0.5, 9.0)
        # self.vb.camera.azimuth = -75.5
        # self.vb.camera.scale_factor = 32.7
        #
        # self.vb_img.camera = 'turntable'
        # self.vb_img.camera.elevation = -90.0
        # self.vb_img.camera.center = (2100, -380, -500)
        # self.vb_img.camera.azimuth = 0.0
        # self.vb_img.camera.scale_factor = 1500

        # @self.canvas.connect
        # def on_key_press(ev):
        #     if ev.key.name in '+=':
        #         a = self.vb.camera.get_state()
        #     print(a)

        # self.input_data(scans)
        #
        # vispy.app.run()

    def input_data(self,scans=None,img=None,boxes=None,index=0,save_img=False,no_gt=False):
        self.canvas = vispy.scene.SceneCanvas(show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.vb = self.grid.add_view(row=0, col=0, row_span=2)
        self.vb_img = self.grid.add_view(row=1, col=0)

        self.vb.camera = 'turntable'
        self.vb.camera.elevation = 90#21.0
        self.vb.camera.center = (6.5, -0.5, 9.0)
        self.vb.camera.azimuth = -90#-75.5
        self.vb.camera.scale_factor = 63#32.7

        self.vb_img.camera = 'turntable'
        self.vb_img.camera.elevation = -90.0
        self.vb_img.camera.center = (2100, -380, -500)
        self.vb_img.camera.azimuth = 0.0
        self.vb_img.camera.scale_factor = 1500

        pos = scans[:, 0:3]
        scatter = visuals.Markers()
        scatter.set_gl_state('translucent', depth_test=False)
        scatter.set_data(pos, edge_width=0, face_color=(1, 1, 1, 1), size=0.01, scaling=True)
        self.vb.add(scatter)

        if img is None:
            img=np.zeros(shape=[1,1,3],dtype=np.float32)
        image = visuals.Image(data=img, method='auto')
        self.vb_img.add(image)

        if boxes is not None:
            if len(boxes.shape)==1:
                boxes = boxes.reshape(1, -1)
            gt_indice = np.where(boxes[:, -1] == 2)[0]
            gt_cnt = len(gt_indice)
            i = 0
            for box in boxes:
                radio = max(box[0] - 0.5, 0.005)*2.0
                color = (0, radio, 0, 1)  # Green

                if box[-1] == 4:  #  gt boxes
                    i = i + 1
                    vsp_box = visuals.Box(width=box[4],  depth=box[5],height=box[6], color=(0.6, 0.8, 0.0, 0.3))#edge_color='yellow')
                    mesh_box = vsp_box.mesh.mesh_data
                    mesh_border_box = vsp_box.border.mesh_data
                    vertices = mesh_box.get_vertices()
                    center = np.array([box[1], box[2], box[3]], dtype=np.float32)
                    vtcs = np.add(vertices, center)
                    mesh_border_box.set_vertices(vtcs)
                    mesh_box.set_vertices(vtcs)
                    self.vb.add(vsp_box)
                    if False:
                        text = visuals.Text(text='gt: ({}/{})'.format(i, gt_cnt), color='white', face='OpenSans', font_size=12,
                                            pos=[box[1], box[2], box[3]],anchor_x='left', anchor_y='top', font_manager=None)
                        self.vb.add(text)

                if (box[-1]+box[-2]) == 0: # True negative cls rpn divided by cube
                    self.vb.add(line_box(box,color=color))
                if (box[-1]+box[-2]) == 1: # False negative cls rpn divided by cube
                    self.vb.add(line_box(box,color='red'))
                if (box[-1]+box[-2]) == 2: # False positive cls rpn divided by cube
                    if no_gt:
                        self.vb.add(line_box(box, color='yellow'))
                    else:
                        self.vb.add(line_box(box, color='blue'))
                if (box[-1]+box[-2]) == 3: # True positive cls rpn divided by cube
                    self.vb.add(line_box(box,color='yellow'))

        if save_img:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fileName = path_add(folder,str(index).zfill(6)+'.png')
            res = self.canvas.render(bgcolor='black')[:,:,0:3]
            vispy_file.write_png(fileName, res)

        @self.canvas.connect
        def on_key_press(ev):
            if ev.key.name in '+=':
                a = self.vb.camera.get_state()
            print(a)
        # vispy.app.run()

    def get_thread_data(self,QUEUE):
        while True:
            if not QUEUE.empty():
                msg = QUEUE.get() # from class msg_qt(object) in file: cubic_train
                scans =msg.scans[0]
                img=msg.img[0]
                boxes_=msg.boxes[0]
                index=msg.index[0]
                save_img=msg.save_img[0]
                no_gt=msg.no_gt[0]
                # win32api.keybd_event(17,0,0,0)
                # pcd_vispy(scans=scans, img=img, boxes=boxes_,index=index,save_img=save_img,
                #           visible=False, no_gt=no_gt, multi_vis=True)
                self.input_data(scans,img,boxes_,index,save_img,no_gt)
                vispy.app.run()
                a =[]

def pcd_vispy(scans=None,img=None, boxes=None, name=None, index=0,vis_size=(800, 600),save_img=False,visible=True,multi_vis=False,point_size=0.02):
    if multi_vis:
        canvas = vispy.scene.SceneCanvas(title=name, keys='interactive', size=vis_size,show=True)
    else:
        canvas = vispy.scene.SceneCanvas(title=name, keys='interactive', size=vis_size,show=visible)
    grid = canvas.central_widget.add_grid()
    vb = grid.add_view(row=0, col=0, row_span=2)
    vb_img = grid.add_view(row=1, col=0)

    vb.camera = 'turntable'
    vb.camera.elevation = 90  # 21.0
    vb.camera.center = (6.5, -0.5, 9.0)
    vb.camera.azimuth = -90  # -75.5
    vb.camera.scale_factor = 63  # 32.7

    if scans is not None:
        if not isinstance(scans, list):
            pos = scans[:, :3]
            scatter = visuals.Markers()
            scatter.set_gl_state('translucent', depth_test=False)
            scatter.set_data(pos, edge_width=0, face_color=(1, 1, 1, 1), size=point_size, scaling=True)
            vb.add(scatter)
        else:
            pos = scans[0][:, :3]
            scatter = visuals.Markers()
            scatter.set_gl_state('translucent', depth_test=False)
            scatter.set_data(pos, edge_width=0, face_color=(1, 1, 1, 1), size=point_size, scaling=True)
            vb.add(scatter)

            pos = scans[1][:, :3]
            scatter = visuals.Markers()
            scatter.set_gl_state('translucent', depth_test=False)
            scatter.set_data(pos, edge_width=0, face_color=(0, 1, 1, 1), size=point_size, scaling=True)
            vb.add(scatter)


    axis = visuals.XYZAxis()
    vb.add(axis)

    if img is None:
        img=np.zeros(shape=[1,1,3],dtype=np.float32)
    image = visuals.Image(data=img, method='auto')
    vb_img.camera = 'turntable'
    vb_img.camera.elevation = -90.0
    vb_img.camera.center = (2100, -380, -500)
    vb_img.camera.azimuth = 0.0
    vb_img.camera.scale_factor = 1500
    vb_img.add(image)

    if boxes is not None:
        gt_indice = np.where(boxes["cls_rpn"] == 4)[0]
        gt_cnt = len(gt_indice)
        boxes_cnt = boxes["center"].shape[0]
        i=0
        for k in range(boxes_cnt):
            radio = max(boxes["score"][k] - 0.5, 0.005)*2.0
            color = (0, radio, 0, 1)  # Green
            if boxes["cls_rpn"][k] == 4:  #  gt boxes
                i = i + 1
                vsp_box = visuals.Box(depth=boxes["size"][k][0],width=boxes["size"][k][1],  height=boxes["size"][k][2], color=(0.3, 0.4, 0.0, 0.06),edge_color='pink')
                mesh_box = vsp_box.mesh.mesh_data
                mesh_border_box = vsp_box.border.mesh_data
                vertices = mesh_box.get_vertices()
                center = np.array([boxes["center"][k][0], boxes["center"][k][1],boxes["center"][k][2]], dtype=np.float32)
                vertices_roa_trans = box_rot_trans(vertices, -boxes["yaw"][k][0], center)#
                mesh_border_box.set_vertices(vertices_roa_trans)
                mesh_box.set_vertices(vertices_roa_trans)
                vb.add(vsp_box)
                if False:
                    text = visuals.Text(text='det: ({}/{})'.format(i, gt_cnt), color='white', face='OpenSans', font_size=12,
                                        pos=[boxes["center"][k][0], boxes["center"][k][1], boxes["center"][k][2]],anchor_x='left', anchor_y='top', font_manager=None)
                    vb.add(text)
            elif (boxes["cls_rpn"][k]+boxes["cls_cube"][k]) == 0: # True negative cls rpn divided by cube
                vb.add(line_box(boxes["center"][k],boxes["size"][k],-boxes["yaw"][k],color=color))
            elif (boxes["cls_rpn"][k]+boxes["cls_cube"][k]) == 1: # False negative cls rpn divided by cube
                vb.add(line_box(boxes["center"][k],boxes["size"][k],-boxes["yaw"][k],color="red"))
            elif (boxes["cls_rpn"][k]+boxes["cls_cube"][k]) == 2: # False positive cls rpn divided by cube
                vb.add(line_box(boxes["center"][k],boxes["size"][k],-boxes["yaw"][k],color="blue"))
            elif (boxes["cls_rpn"][k]+boxes["cls_cube"][k]) == 3: # True positive cls rpn divided by cube
                vb.add(line_box(boxes["center"][k],boxes["size"][k],-boxes["yaw"][k],color="yellow"))
            text = visuals.Text(text=str(k), color=color, face='OpenSans', font_size=12,
                                pos=[boxes["center"][k][0]-boxes["size"][k][0]/2, boxes["center"][k][1]-boxes["size"][k][1]/2, boxes["center"][k][2]-boxes["size"][k][2]/2], anchor_x='left', anchor_y='top', font_manager=None)

            vb.add(text)

    if save_img:
        folder = path_add(cfg.TEST_RESULT, cfg.RANDOM_STR)
        if not os.path.exists(folder):
            os.makedirs(folder)
        fileName = path_add(folder,str(index).zfill(6)+'.png')
        res = canvas.render(bgcolor='black')[:,:,0:3]
        vispy_file.write_png(fileName, res)

    @canvas.connect
    def on_key_press(ev):
        if ev.key.name in '+=':
            a = vb.camera.get_state()
        print(a)

    if visible:
        pass
        vispy.app.run()

    return canvas

def box_rot_trans(vertices, rotation,translation):
    # points: numpy array;translation: moving scalar which should be small
    R = np.array([[np.cos(rotation), -np.sin(rotation), 0.],
                  [np.sin(rotation), np.cos(rotation), 0.],
                  [0, 0, 1]], dtype=np.float32)
    translation = np.reshape(translation,[3,1])
    points_rot = np.add(np.matmul(R, vertices.transpose()),translation)

    return points_rot.transpose()

def pcd_show_now():
    vispy.app.run()
    vispy.app.quit()

def vispy_init():
    import vispy.app
    # vispy.use('pyqt4')
    # vispy.app.use_app()
    v = vispy.app.Canvas()

def line_box(box_center,box_size,rot,color=(0, 1, 0, 0.1)):
    box = np.array([box_center[0],box_center[1],box_center[2],box_size[1],box_size[0],box_size[2]],dtype=np.float32)#box_size[1] #TODO:just for view
    p0, p1, p2, p3, p4, p5, p6, p7=box3d_2conner(box,rot)
    pos = np.vstack((p0,p1,p2,p3,p0,p4,p5,p6,p7,p4,p5,p1,p2,p6,p7,p3))
    lines = visuals.Line(pos=pos, connect='strip', width=1, color=color, antialias=True,method='gl')

    return lines

#  using RViz  ===========================

def Boxes_labels_Gen(box_es,ns,frame_id='rslidar'):
    from visualization_msgs.msg import Marker,MarkerArray
    from geometry_msgs.msg import Point,Vector3,Quaternion
    from std_msgs.msg import ColorRGBA

    def one_box(box_,color,index):
        marker = Marker()
        marker.id = index
        marker.ns= ns
        marker.header.frame_id = frame_id
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        # marker.frame_locked=False
        # marker scale
        marker.scale = Vector3(0.07, 0.07, 0.07)  # x,yz
        # marker color
        marker.color = ColorRGBA(color[0], color[1], color[2], color[3])  # r,g,b,a
        # marker orientaiton
        marker.pose.orientation = Quaternion(0., 0., 0., 1.)  # x,y,z,w
        # marker position
        marker.pose.position = Point(0., 0., 0.)  # x,y,z
        # marker.lifetime = rospy.Duration(0.1)
        p0, p1, p2, p3, p4, p5, p6, p7 = box3d_2conner(box_,0)
        # marker line points
        marker.points = []
        for p in [p0, p1, p2, p3, p0, p4, p5, p6, p7, p4, p5, p1, p2, p6, p7, p3]:
            marker.points.append(Point(p[0], p[1], p[2], ))

        return marker

    def delete_all_markers(box_,color,index):
        marker = Marker()
        marker.id = index
        marker.ns = ns
        marker.header.frame_id = frame_id
        marker.type = marker.LINE_STRIP
        marker.action = 3 # marker.DELETEALL: deletes all objects
        # marker.frame_locked=False
        # marker scale
        marker.scale = Vector3(0.04, 0.04, 0.04)  # x,yz
        # marker color
        marker.color = ColorRGBA(color[0], color[1], color[2], color[3])  # r,g,b,a
        # marker orientaiton
        marker.pose.orientation = Quaternion(0., 0., 0., 1.)  # x,y,z,w
        # marker position
        marker.pose.position = Point(0., 0., 0.)  # x,y,z
        # marker.lifetime = rospy.Duration(0.1)
        p0, p1, p2, p3, p4, p5, p6, p7 = box3d_2conner(box_,0)
        # marker line points
        marker.points = []
        for p in [p0, p1, p2, p3, p0, p4, p5, p6, p7, p4, p5, p1, p2, p6, p7, p3]:
            marker.points.append(Point(p[0], p[1], p[2], ))

        return marker

    label_boxes = MarkerArray()
    label_boxes.markers=[]
    # TODO: to fix boxes type from array2dict
    for idx,_box in enumerate(box_es):
        radio = max(_box[6] - 0.5, 0.005) * 2.0
        color = (0, radio, 0, 1)  # Green
        if _box[-1]==1:
            color_ = (1., 1., 0., 1)  # yellow
        else:
            color_ = color  # green
        if idx == 0:
            label_boxes.markers.append(delete_all_markers(_box, color_, idx))
        label_boxes.markers.append(one_box(_box,color_,idx))

    return label_boxes

def PointCloud_Gen(points,frameID='rslidar'):
    from sensor_msgs.msg import PointCloud, ChannelFloat32
    from geometry_msgs.msg import Point32

    ##=========PointCloud===============
    points.dtype = np.float32
    point_cloud = points.reshape((-1, 4))
    pointx = point_cloud[:, 0].flatten()
    pointy = point_cloud[:, 1].flatten()
    pointz = point_cloud[:, 2].flatten()
    intensity = point_cloud[:, 3].flatten()
    # labels = point_cloud[:,6].flatten()

    seg_point = PointCloud()
    seg_point.header.frame_id = frameID
    channels1 = ChannelFloat32()
    seg_point.channels.append(channels1)
    seg_point.channels[0].name = "rgb"
    channels2 = ChannelFloat32()
    seg_point.channels.append(channels2)
    seg_point.channels[1].name = "intensity"

    for i in range(point_cloud.shape[0]):
        seg_point.channels[1].values.append(intensity[i])
        if True:  # labels[i] == 1:
            seg_point.channels[0].values.append(255)
            geo_point = Point32(pointx[i], pointy[i], pointz[i])
            seg_point.points.append(geo_point)
        else:
            seg_point.channels[0].values.append(255255255)
            geo_point = Point32(pointx[i], pointy[i], pointz[i])
            seg_point.points.append(geo_point)
            # elif result[i] == 2:
            #     seg_point.channels[0].values.append(255255255)
            #     geo_point = Point32(pointx[i], pointy[i], pointz[i])
            #     seg_point.points.append(geo_point)
            # elif result[i] == 3:
            #     seg_point.channels[0].values.append(255000)
            #     geo_point = Point32(pointx[i], pointy[i], pointz[i])
            #     seg_point.points.append(geo_point)

    return seg_point

def Image_Gen(iamge,frameID='rslidar'):
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    cv_ros = CvBridge()
    image_ros = cv_ros.cv2_to_imgmsg(iamge,'bgr8')
    # res = Image()
    # image_ros.data = iamge
    # image_ros.height = 375
    # image_ros.width = 1242
    image_ros.header.frame_id = frameID
    return image_ros

#  using mayavi ===========================

def draw_3dPoints_box(lidar=None, Boxes3D=None, is_grid=True, fig=None, draw_axis=True):
    import mayavi.mlab as mlab  # 3d point

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
        pass

    if lidar is not None:
        mlab.points3d(pxs, pys, pzs, prs,
                      mode='point',  # 'point'  'sphere'
                      colormap='gnuplot',  # 'bone',  #'spectral',  #'copper',
                      scale_factor=1,
                      figure=fig)

    if Boxes3D is not None:
        for i in range(Boxes3D.shape[0]):
            b = lidar_3d_to_corners(Boxes3D[i, 1:7].reshape(-1, 6)).reshape(3, 8).transpose()
            a = round(Boxes3D[i, 0], 2)
            if a == 1.0:
                mycolor = (0., 1., 0.)
            else:
                a = max(a - 0.6, 0.025) * 2.5 + 0.01
                mycolor = (a, a, a)

            for k in range(0, 4):
                # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
                i, j = k, (k + 1) % 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=mycolor, tube_radius=None,
                            line_width=1, figure=fig)

                i, j = k + 4, (k + 1) % 4 + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=mycolor, tube_radius=None,
                            line_width=1, figure=fig)

                i, j = k, k + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=mycolor, tube_radius=None,
                            line_width=1, figure=fig)

    # draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        for y in np.arange(-40, 40, 5):
            x1, y1, z1 = -40.0, float(y), -1.5
            x2, y2, z2 = 40.0, float(y), -1.5
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.1, 0.1, 0.1), tube_radius=None, line_width=0.1,
                        figure=fig)

        for x in np.arange(-40, 40, 5):
            x1, y1, z1 = float(x), -40.0, -1.5
            x2, y2, z2 = float(x), 40.0, -1.5
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.1, 0.1, 0.1), tube_radius=None, line_width=0.1,
                        figure=fig)

    # draw axis
    if draw_axis:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        axes = np.array([
            [2., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 2., 0.],
        ], dtype=np.float64)
        fov = np.array([
            [40., 40., 0., 0.],
            [40., -40., 0., 0.],
        ], dtype=np.float64)

        mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                    figure=fig)
        mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                    figure=fig)

    mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=None, distance=50, focalpoint=[12.0909996, -1.04700089, -2.03249991])

    mlab.show()

#  using tensorboard ===========================
def show_rpn_tf(img, cls_bv):  # TODO
    bv_data = tf.reshape(img[:, :, :, 8], (601, 601, 1))
    bv_data = scales_to_255(bv_data, 0, 3, tf.float32)
    bv_img = tf.reshape(tf.stack([bv_data, bv_data, bv_data], 3), (601, 601, 3))

    return tf.py_func(show_bbox, [bv_img, cls_bv], tf.float32)

def show_bbox(bv_image, cls_bv):
    cnt = cls_bv.shape[0]
    for i in range(cnt):
        if cls_bv[i, 4] == 1:
            cv2.rectangle(bv_image, (cls_bv[i, 0], cls_bv[i, 1]), (cls_bv[i, 2], cls_bv[i, 3]), color=(60, 60, 0))
        else:
            cv2.rectangle(bv_image, (cls_bv[i, 0], cls_bv[i, 1]), (cls_bv[i, 2], cls_bv[i, 3]), color=(0, 30, 0))
    # filePath = "/media/disk4/deeplearningoflidar/he/CombiNet-he/output/"
    # cv2.imwrite(filePath+fileName,bv_image)
    return bv_image

def test_show_rpn_tf(img, box_pred=None):
    bv_data = tf.reshape(img[:, :, :, 8],(601, 601, 1))
    bv_data = scales_to_255(bv_data,0,3,tf.float32)
    bv_img = tf.reshape(tf.stack([bv_data,bv_data,bv_data],3),(601,601,3))
    return tf.py_func(test_show_bbox, [bv_img,box_pred], tf.float32)

def test_show_bbox(bv_image, bv_box):
    for i in range(bv_box.shape[0]):
        a = bv_box[i, 0]*255
        color_pre = (a, a, a)
        cv2.rectangle(bv_image, (bv_box[i, 1], bv_box[i, 2]), (bv_box[i, 3], bv_box[i, 4]), color=color_pre)

    return bv_image


if __name__ == '__main__':
    # import rospy
    # from visualization_msgs.msg import Marker, MarkerArray
    #
    # boxes =np.array([[1,1,1,1,1,1,1,1,1],[1,3,3,1,1,1,1,1,1]])
    # rospy.init_node('node_labels')
    # label_pub = rospy.Publisher('labels', MarkerArray, queue_size=100)
    # rospy.loginfo('Ros begin ...')
    # label_box = Boxes_labels_Gen(boxes,ns='test_box')
    # while True:
    #     label_pub.publish(label_box)
    ##====================================================================
    from multiprocessing import Process, Queue
    from dataset.dataset import dataset_KITTI_train
    from easydict import EasyDict as edict
    from cubicnet.cubic_train import msg_qt
    import time
    import virtkey

    arg = edict()
    arg.imdb_type = 'kitti'
    arg.method = 'train'

    dataset = dataset_KITTI_train(arg)
    blobs = dataset.get_minibatch(169)
    MSG_QUEUE = Queue(200)
    station = pcd_vispy_client(MSG_QUEUE,blobs['lidar3d_data'],title='Vision')

    vision_qt = Process(target=station.get_thread_data, args=(MSG_QUEUE,))
    vision_qt.start()
    while True:
        msg = msg_qt(scans=blobs['lidar3d_data'], boxes= blobs['gt_boxes_3d'], name='CubicNet training')
        MSG_QUEUE.put(msg)
        time.sleep(4)
