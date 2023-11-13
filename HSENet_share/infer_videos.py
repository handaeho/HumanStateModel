from detectron2.engine import DefaultPredictor
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch

import cv2
from PIL import Image
import time
import glob
import os
import tqdm
import numpy as np
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import (
    COCO_PERSON_KEYPOINT_NAMES,
    COCO_PERSON_KEYPOINT_FLIP_MAP,
    KEYPOINT_CONNECTION_RULES
)
from detectron2 import model_zoo

# HSENet_example
from hsenet.modeling.roi_heads.roi_heads import HSENetROIHeads
#from hsenet.config.config import add_posture_config
from hsenet.utils.visualizer import Visualizer

meta = {
        "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
        "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
        "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
        "thing_classes": ["Walking", "Crouch", "Lying", "Standing", "Running", "Sitting"],
        "thing_colors": [(0,0,0), (0,0,0), (0,0,255), (0,0,0), (0,0,0), (0,0,0)],
        }

for k in meta.keys():
    eval('MetadataCatalog.get("test").set({}=meta[k])'.format(k))
data_meta = MetadataCatalog.get("test")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    #add_posture_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--output_dir", default='results', type=str)
    parser.add_argument("--video", required=True, type=str)
    args = parser.parse_args()
    print("Command Line Args:", args)

    # Inference with a multitask model
    cfg = setup(args)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # set threshold for this model


    predictor = DefaultPredictor(cfg)

    vis = True
    save_video = False
    video_resolution = None
    #video_resolution = (1280, 720)
    #video_resolution = (640, 360)
    #video_resolution = (320, 180)

    if os.path.isdir(args.video):
        video_list = glob.glob(args.video + '/*')
    else:
        video_list = [args.video]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    save_path_video = args.output_dir+"_video/" + video_list[0].split('/')[-2]
    if not os.path.exists(save_path_video):
        os.makedirs(save_path_video)


    with torch.no_grad():
        pbar = tqdm.tqdm(video_list, dynamic_ncols=True)
        for video_name in pbar:
            cap = cv2.VideoCapture(video_name)
            frame_count = 0
            if not cap.isOpened():
                raise Exception('Invalid video cannot open video {}!!!!'.format(video_name))

            # define save video name
            if save_video:
                write_name_video = os.path.basename(video_name) + '.mp4'
                # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')

                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if not video_resolution:
                    video_resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                out_video = cv2.VideoWriter(save_path_video + '/' + write_name_video, fourcc, fps, video_resolution)

            while 1:
                # Read frames
                ret, im = cap.read()
                if not ret:
                    break
                if video_resolution:
                    im = cv2.resize(im, video_resolution)
                # Inference and measure time
                t_1 = time.time()
                outputs = predictor(im)
                t_2 = time.time() - t_1
                #print("Inference time: {:0.4f}, FPS: {:.4f}".format(t_2, 1/t_2), end='\r')
                pbar.set_description("Inference: {:0.2f} ms, FPS: {:.2f}".format(t_2*1000, 1/t_2))
                #pbar.set_description("Latency: {:0.4f}, FPS: {:.4f}".format(t_2, 1/t_2))

                frame_count+=1

                if (outputs["instances"].pred_classes==2).sum():
                    ind = (outputs["instances"].pred_classes==2).long().argsort(descending=True)
                    outputs["instances"] = outputs["instances"][ind]

                if vis or save_video:
                    v = Visualizer(im[:,:,::-1], data_meta, scale=1.2, instance_mode=ColorMode.SEGMENTATION)
                    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                if vis:
                    cv2.imshow('video', out.get_image()[:,:,::-1])
                    cv2.waitKey(1)
                if save_video:
                    out_frame = cv2.resize(out.get_image()[:,:,::-1], video_resolution)
                    out_video.write(out_frame)
