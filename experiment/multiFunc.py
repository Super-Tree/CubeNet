import numpy as np
import cv2
# import rospy
# from sensor_msgs.msg import PointCloud
import os
import matplotlib.pyplot as plt
import random
import mayavi.mlab as mlab  # 3d point
import cPickle as pickle

from sensor_msgs.msg import PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32

import os.path as osp
import sys

def label_convertor(path,name):
    file_sorted=sorted(os.listdir(path))

    with open(name,'w') as det_txt_fid:
        for file in file_sorted:
            with open(os.path.join(path,file)) as one_det:
                lines= one_det.readlines()
                for line in lines:
                    line=line.replace("-1 -1 -10","-1.00")
                    line=line.replace("Cyclist","2")
                    line=line.replace("Car","1")
                    line=line.replace("Pedestrian","3")
                    line=line.replace("\n"," 1\n")
                    modify_line=file[0:6]+".png "+line
                    det_txt_fid.write(modify_line)


if __name__ == '__main__':
    path='/home/likewise-open/SENSETIME/hexindong/ProjectDL/Evalue3d/det_files/pre-traindet_files'
    label_convertor(path,"/home/likewise-open/SENSETIME/hexindong/ProjectDL/Evalue3d/det_files/det_pre-train.txt")
    path2='/home/likewise-open/SENSETIME/hexindong/ProjectDL/Evalue3d/det_files/re-traindet_files'
    label_convertor(path2,"/home/likewise-open/SENSETIME/hexindong/ProjectDL/Evalue3d/det_files/det_re-train.txt")