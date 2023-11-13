import timeit
import cv2
import time
import torch
import multiprocessing

from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, \
    KEYPOINT_CONNECTION_RULES
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
from HSENet.HSENet import vms


# TODO: GPU 0번에 올라가 연산 될 1번 RTSP 그룹 -> 카메라 채널 1번 ~ 5번

