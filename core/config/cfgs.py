import tensorflow as tf
import tensorflow.contrib.slim as slim
# training

TRAIN_DATAETS_DIR = r'F:\Datasets\obj_det\vidrone\visdrone2020\VisDrone2019-DET-train'
DATASETS_NAME = 'visdrone'
CLASS_NUM = 12
USE_SUMMARY = False
SUMMARY_STEP = 200
SUMMARY_DIR = "./log/"
SAVED_STEP = 1000
SAVED_WEIGHTS_DIR = "./saved_weights/"
PRETRAIN_BACKBONE_WEIGHTS = r"./data/pretrained_weights/resnet50_v1d.ckpt"
PRETRAIN_BACKBONE_NAME = 'resnet50_v1d' #resnet50_v1d, resnet101_v1d

#
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_REGULARIZER= slim.l2_regularizer(0.0001)

LR = 0.002
DECAY_STEP = [70000, 90000]
SHORT_SIZE_LIMITATION = 800
SHORT_SIZE_LIMITATION_LIST=[600,800,1000,1200]
MAX_LENGTH = 10000  # set no limit for long side of image
MAX_ITERATION = 90000
PIXEL_MEAN = [123.68, 116.779, 103.939]
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]
MOMENTUM = 0.9
FREEZE_BLOCKS = [True, False, False, False, False]  # resnet backbone
# Anchor
LEVELS = ['P2', 'P3', 'P4', 'P5', 'P6']
# can be modified according to datasets
BASE_ANCHOR_SIZE_LIST = [16, 32, 64, 128, 256]
ANCHOR_STRIDE_LIST = [4, 8, 16, 32, 64]
ANCHOR_RATIOS = [0.5, 1., 2.0]
ROI_SCALE_FACTORS = [[10., 10., 5.0, 5.0], [20., 20., 10.0, 10.0], [30., 30., 15.0, 15.0]]
ANCHOR_SCALE_FACTORS = None  # [10., 10., 5.0, 5.0]
IS_FILTER_OUTSIDE_BOXES = True
# RPN
RPN_CLS_LOSS_WEIGHT = 1.0
RPN_BBOX_LOSS_WEIGHT = 1.0
RPN_SIGMA_FACTOR = 3.0
TRAIN_RPN_CLOOBER_POSITIVES = False
RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THR = 0.7
RPN_PRE_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000
RPN_PRE_NMS_TEST = 6000
RPN_MAXIMUM_PROPOSAL_TEST = 1500
# fast rcnn
CASCADE_SAMPLE_IOU_THR=[0.5,0.6,0.7]
RCNN_BBOX_LOSS_WEIGHT = 1.0
RCNN_CLS_LOSS_WEIGHT = 1.0
RCNN_SIGMA_FACTOR = 1.0
SUMMARY_SCORE_THR = 0.4
RCNN_NMS_IOU_THR = 0.5 #
RCNN_NMS_MAX_BOXES_PER_CLASS = 100
RCNN_IOU_POSITIVE_THR = 0.5
RCNN_IOU_NEGATIVE_THR = 0.0
RCNN_MINIBATCH_SIZE = 512
RCNN_POSITIVE_RATE = 0.25

# tricks
EXTRA_CONV_FOR_REG = 0 # double head
MULTI_SCALSE_TRAINING=False # multi-scale training
GLOBAL_CTX = True

# eval
INFERENCE_SCORE_THRSHOLD = 0.4
IMG_INFERENCE_DIR = r'F:\Datasets\obj_det\vidrone\visdrone2020\VisDrone2019-DET-val\images'
MULTI_SCALSE_TESTING=False
NORMALIZED_RESULTS_FOR_MODEL_ENSEMBLE=False # set True to get normailized coords result
