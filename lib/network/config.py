# coding=utf-8
import os.path as osp
from easydict import EasyDict as edict
from distutils import spawn
import random
import string
import os
import numpy as np
import socket
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0':log, '1':info, '2':warning ,'3':Error} #TODO : check
__C = edict()

cfg = __C

__C.GPU_AVAILABLE = '3,1,2,0'

__C.NUM_CLASS = 2
__C.DEFAULT_PADDING = 'SAME'
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

if socket.gethostname()=='szstdzcp0325':
    __C.DATA_DIR = "/home/likewise-open/SENSETIME/hexindong/ProjectDL/cubic-local/DATASET/KITTI/object"
    __C.DATA_BOXES_DIR = "/home/likewise-open/SENSETIME/hexindong/ProjectDL/cubic-local/DATASET/KITTI/object/box_car_only"
else:
    __C.DATA_DIR ="/mnt/lustre/hexindong/ProjectDL/CubeNet-server/DATASET/KITTI/object"
    __C.DATA_BOXES_DIR = "/mnt/lustre/hexindong/ProjectDL/CubeNet-server/DATASET/KITTI/object/box_car_only"

__C.DATA_HANGZHOU_DIR =osp.join(__C.ROOT_DIR,'DATASET','Hangzhou','2018-07-06-16-21-48')


__C.OUTPUT_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'output'))
__C.LOG_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'log'))
__C.LOCAL_LOG_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'local_log'))
__C.TEST_RESULT = osp.abspath(osp.join(__C.ROOT_DIR, 'test_result'))
__C.EPS = 1e-15  # 1e-3
__C.ANCHOR = [4.000,4.000,2.000]  # car size # todo: car height should be carefully decided!
__C.CUBIC_RES = [0.136,0.136,0.14] # 30x30x15  # car size [0.2858,0.2858,0.1429]:14x14x14
__C.CUBIC_SIZE = [int(np.ceil(np.round(__C.ANCHOR[i] / __C.CUBIC_RES[i], 3))) for i in range(3)] # Be careful about python number decimal
__C.ANCHOR_CNT=1
__C.RPN_POINTS_REMAIN = 600
__C.VOXEL_POINT_COUNT = 20
# TODO:DETECTION_RANGE should be careful !
# effect on GroundTruth range filter(dataset_STI_train.filter) and  pseudo-rpn generate in function 'proposal_layer_3d_STI'
# when change the detection range ,must delete the cache file in dataset
__C.DETECTION_RANGE = 20.0
__C.RANDOM_STR =''.join(random.sample(string.uppercase, 4))
if spawn.find_executable("nvcc",path="/usr/local/cuda-8.0/bin/"):
    # Use GPU implementation of non-maximum suppression
    __C.USE_GPU_NMS = True

    # Default GPU device id
    __C.GPU_ID = 0
else:
    print ('File: config.py '
           'Notice: nvcc not found')
    __C.USE_GPU_NMS = False


# Training options
__C.TRAIN = edict()

__C.TRAIN.SNAPSHOT_ITERS = 8000
__C.TRAIN.LEARNING_RATE = 1e-3
__C.TRAIN.BATCH_SIZE = 1  # only one image
__C.TRAIN.WEIGHT_DECAY = 0.0005 # for l2 regularizer in network.py btw,useless
__C.TRAIN.MOVING_AVERAGE_DECAY = 0.9999
__C.TRAIN.ITER_DISPLAY = 10
__C.TRAIN.FOCAL_LOSS = True
__C.TRAIN.TENSORBOARD = True
__C.TRAIN.EPOCH_MODEL_SAVE = True
__C.TRAIN.DEBUG_TIMELINE = True  # Enable timeline generation
__C.TRAIN.USE_VALID = True
__C.TRAIN.VISUAL_VALID = True
__C.TRAIN.USE_AUGMENT_IN_CUBIC_GEN = True
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 128
# Max number of foreground examples ,only keep 1/4 positive anchors
__C.TRAIN.RPN_FG_FRACTION = 0.25
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.75
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.4
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False


# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 4000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 50
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.5


# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.7
# Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1


# Testing options
__C.TEST = edict()
__C.TEST.ITER_DISPLAY = 1
__C.TEST.SAVE_IMAGE = False
# NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.32

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 3000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 50
__C.TEST.TENSORBOARD = True
__C.TEST.DEBUG_TIMELINE = False
