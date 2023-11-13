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
#from hsenet.modeling.roi_heads.roi_heads import HSENetROIHeads

#from hsenet.config.config import add_posture_config
#from hsenet.data import DatasetMapper, build_detection_train_loader#, build_detection_test_loader

dataset_root = './datasets/'

meta = {
        "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
        "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
        "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
        }

register_coco_instances("MPHBE2020_train", meta, dataset_root + "MPHBE2020/annotations/instances_mphbE_train2020.json", dataset_root + "MPHBE2020/train2020")
register_coco_instances("MPHBE2020_test", meta, dataset_root + "MPHBE2020/annotations/instances_mphbE_test2020.json", dataset_root + "MPHBE2020/test2020")
register_coco_instances("MPHBE2021_train", meta, dataset_root + "MPHBE2021/annotations/instances_mphbE_train2021.json", dataset_root + "MPHBE2021/train2021")
register_coco_instances("MPHBE2021_test", meta, dataset_root + "MPHBE2021/annotations/instances_mphbE_test2021.json", dataset_root + "MPHBE2021/test2021")
register_coco_instances("keypoints_coco_2017_train_mphbe", meta, dataset_root + "coco/annotations/person_keypoints_train2017.json", dataset_root + "coco/train2017")
register_coco_instances("keypoints_coco_2017_val_mphbe", meta, dataset_root + "coco/annotations/person_keypoints_val2017.json", dataset_root + "coco/val2017")

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
    args = default_argument_parser()
    args.add_argument('--img', default=None)
    args = args.parse_args()
    print("Command Line Args:", args)

    # Inference with a multitask model
    cfg = setup(args)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # set threshold for this model

    predictor = DefaultPredictor(cfg)

    if os.path.isdir(args.img):
        im_list = glob.glob(args.img + '/*')
    else:
        im_list = [args.img]


    print(fvcore.nn.parameter_count_table(predictor.model))
    data_meta = MetadataCatalog.get("MPHBE2020_test").set(thing_classes=["Walking", "Crouch", "Lying", "Standing", "Running", "Sitting"])

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

        img.save('test_results/{}'.format(os.path.basename(im_)))

        #cv2.imshow('test', out.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
