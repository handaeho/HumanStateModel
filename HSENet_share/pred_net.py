from detectron2.engine import DefaultPredictor
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

import torch
import cv2
from PIL import Image
from detectron2.utils.visualizer import Visualizer
import fvcore
import os
import glob
import tqdm
import time

from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, KEYPOINT_CONNECTION_RULES
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

# HSENet_example

#from hsenet.data.datasets import register_coco_instances
from hsenet.modeling.roi_heads.roi_heads import HSENetROIHeads
from hsenet.utils.visualizer import Visualizer
from hsenet.config.config import add_posture_config
import hsenet.modeling.backbone

dataset_root = './datasets/'

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
    add_posture_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument('--img', default=None)
    args = args.parse_args()
    print("Command Line Args:", args)

    # Inference with a multitask model
    cfg = setup(args)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # set threshold for this model

    predictor = DefaultPredictor(cfg)

    if os.path.isdir(args.img):
        im_list = glob.glob(args.img + '/*')
    else:
        im_list = [args.img]

    test_result = './results'
    if not os.path.exists(test_result):
        os.makedirs(test_result)

    print(fvcore.nn.parameter_count_table(predictor.model))

    t_bar = tqdm.tqdm(im_list)
    for im_ in t_bar:
        im = cv2.imread(im_)
        t = time.time()
        outputs = predictor(im)
        t_bar.set_description("Inference: {:.1f}ms".format(1000*(time.time() - t)))
        t_bar.refresh()

        v = Visualizer(im[:,:,::-1], data_meta, scale=1.2)

        #outputs["instances"].remove('pred_classes')
        #outputs["instances"].remove('pred_boxes')
        #outputs["instances"].remove('scores')
        #outputs["instances"].remove('pred_keypoints')
        #outputs["instances"].remove('pred_masks')
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = Image.fromarray(out.get_image())
        #img.save('result1.jpg')

        if len(im_list)<2:
            img.show()

        img.save('{}/{}'.format(test_result, os.path.basename(im_)))

        #cv2.imshow('test', out.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
