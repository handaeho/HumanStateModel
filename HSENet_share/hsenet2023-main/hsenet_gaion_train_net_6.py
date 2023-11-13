#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
from collections.abc import Mapping

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    DatasetEvaluator,
    inference_on_dataset,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.datasets.builtin_meta import (
    COCO_PERSON_KEYPOINT_NAMES,
    COCO_PERSON_KEYPOINT_FLIP_MAP,
    KEYPOINT_CONNECTION_RULES,
)
from hsenet.config.config import add_posture_config

# HSENet
from hsenet.data.datasets import register_coco_instances
from hsenet.modeling.roi_heads.roi_heads import HSENetROIHeads
import hsenet.modeling.backbone

# from hsenet.config.config import add_posture_config
from hsenet.data import (
    DatasetMapper,
    build_detection_train_loader,
)  # , build_detection_test_loader
from hsenet_coco_evaluation import HsenetCOCOEvaluator

""" Gaion(Tiep) Version Train With gaion_ver_train_270000_0_02.yaml """

meta = {
    "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
    "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
    "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
    # "thing_classes": ['Walking', 'Crouch', 'Lying', 'Standing', 'Running', 'Sitting'],
}

root_path = "/gpu-home/HSENet_dataset/"
train_path = root_path + "train_datasets/"
test_path = root_path + "test_datasets/"

""" Train Datasets """
# ETRI Train Datasets
register_coco_instances(name="ETRI_Lying_train", metadata=meta,
                        json_file=train_path + "etri/ETRI_Lying/annotations/instances_ihp2021_8.json",
                        image_root=train_path + "etri/ETRI_Lying/images", dataset_source=0)
register_coco_instances(name="IHPE_train", metadata=meta,
                        json_file=train_path + "etri/IHPE/annotations/instances_mphbE_train2021.json",
                        image_root=train_path + "etri/IHPE/train2021", dataset_source=0)

# OpenAI Train Datasets
register_coco_instances(name="openai_child_train", metadata=meta,
                        json_file=train_path + "openai/openai_child_train/annotations/openai_child_train.json",
                        image_root=train_path + "openai/openai_child_train/images", dataset_source=0)
register_coco_instances(name="openai_senior_train", metadata=meta,
                        json_file=train_path + "openai/openai_senior_train/annotations/openai_senior_train.json",
                        image_root=train_path + "openai/openai_senior_train/images", dataset_source=0)
register_coco_instances(name="openai_subway_train", metadata=meta,
                        json_file=train_path + "openai/openai_subway_train/annotations/openai_subway_train.json",
                        image_root=train_path + "openai/openai_subway_train/images", dataset_source=0)

# Directing Train Datasets
register_coco_instances(name="directing_asem_train", metadata=meta,
                        json_file=train_path + "directing/directing_asem_train/annotations/directing_asem_train.json",
                        image_root=train_path + "directing/directing_asem_train/images", dataset_source=0)
register_coco_instances(name="directing_halles_train", metadata=meta,
                        json_file=train_path + "directing/directing_halles_train/annotations/directing_halles_train.json",
                        image_root=train_path + "directing/directing_halles_train/images", dataset_source=0)
register_coco_instances(name="directing_liveplazaes_train", metadata=meta,
                        json_file=train_path + "directing/directing_liveplazaes_train/annotations/directing_liveplazaes_train.json",
                        image_root=train_path + "directing/directing_liveplazaes_train/images", dataset_source=0)
register_coco_instances(name="directing_post_train", metadata=meta,
                        json_file=train_path + "directing/directing_post_train/annotations/directing_post_train.json",
                        image_root=train_path + "directing/directing_post_train/images", dataset_source=0)
register_coco_instances(name="directing_west_train", metadata=meta,
                        json_file=train_path + "directing/directing_west_train/annotations/directing_west_train.json",
                        image_root=train_path + "directing/directing_west_train/images", dataset_source=0)
register_coco_instances(name="directing_donggyeong_train", metadata=meta,
                        json_file=train_path + "directing/directing_donggyeong_train/annotations/directing_donggyeong_train.json",
                        image_root=train_path + "directing/directing_donggyeong_train/images", dataset_source=0)
register_coco_instances(name="directing_hallinside_train", metadata=meta,
                        json_file=train_path + "directing/directing_hallinside_train/annotations/directing_hallinside_train.json",
                        image_root=train_path + "directing/directing_hallinside_train/images", dataset_source=0)
register_coco_instances(name="directing_liveplazaif_train", metadata=meta,
                        json_file=train_path + "directing/directing_liveplazaif_train/annotations/directing_liveplazaif_train.json",
                        image_root=train_path + "directing/directing_liveplazaif_train/images", dataset_source=0)
register_coco_instances(name="directing_shinhan_train", metadata=meta,
                        json_file=train_path + "directing/directing_shinhan_train/annotations/directing_shinhan_train.json",
                        image_root=train_path + "directing/directing_shinhan_train/images", dataset_source=0)
register_coco_instances(name="directing_zaraes_train", metadata=meta,
                        json_file=train_path + "directing/directing_zaraes_train/annotations/directing_zaraes_train.json",
                        image_root=train_path + "directing/directing_zaraes_train/images", dataset_source=0)
register_coco_instances(name="directing_frontofstore_train", metadata=meta,
                        json_file=train_path + "directing/directing_frontofstore_train/annotations/directing_frontofstore_train.json",
                        image_root=train_path + "directing/directing_frontofstore_train/images", dataset_source=0)
register_coco_instances(name="directing_linko_train", metadata=meta,
                        json_file=train_path + "directing/directing_linko_train/annotations/directing_linko_train.json",
                        image_root=train_path + "directing/directing_linko_train/images", dataset_source=0)
register_coco_instances(name="directing_megabox1f_train", metadata=meta,
                        json_file=train_path + "directing/directing_megabox1f_train/annotations/directing_megabox1f_train.json",
                        image_root=train_path + "directing/directing_megabox1f_train/images", dataset_source=0)
register_coco_instances(name="directing_zarahome_train", metadata=meta,
                        json_file=train_path + "directing/directing_zarahome_train/annotations/directing_zarahome_train.json",
                        image_root=train_path + "directing/directing_zarahome_train/images", dataset_source=0)
register_coco_instances(name="directing_frontoftwee_train", metadata=meta,
                        json_file=train_path + "directing/directing_frontoftwee_train/annotations/directing_frontoftwee_train.json",
                        image_root=train_path + "directing/directing_frontoftwee_train/images", dataset_source=0)
register_coco_instances(name="directing_liveplazab2_train", metadata=meta,
                        json_file=train_path + "directing/directing_liveplazab2_train/annotations/directing_liveplazab2_train.json",
                        image_root=train_path + "directing/directing_liveplazab2_train/images", dataset_source=0)
register_coco_instances(name="directing_megaboxb1_train", metadata=meta,
                        json_file=train_path + "directing/directing_megaboxb1_train/annotations/directing_megaboxb1_train.json",
                        image_root=train_path + "directing/directing_megaboxb1_train/images", dataset_source=0)
register_coco_instances(name="directing_starbucks_train", metadata=meta,
                        json_file=train_path + "directing/directing_starbucks_train/annotations/directing_starbucks_train.json",
                        image_root=train_path + "directing/directing_starbucks_train/images", dataset_source=0)
register_coco_instances(name="directing_smartphone_train", metadata=meta,
                        json_file=train_path + "directing/directing_smartphone_train/annotations/directing_smartphone_train.json",
                        image_root=train_path + "directing/directing_smartphone_train/images", dataset_source=0)

""" Test Datasets """
# ETRI Test Dataset
register_coco_instances(name="IHPE_test", metadata=meta,
                        json_file=test_path + "etri/IHPE/annotations/instances_mphbE_test2021.json",
                        image_root=test_path + "etri/IHPE/test2021", dataset_source=0)

# # Directing + OpenAI Dataset
# register_coco_instances(name="directing_openai_test", metadata=meta,
#                         json_file=test_path + "directing_openai/annotations/directing_openai.json",
#                         image_root=test_path + "directing_openai/images", dataset_source=0)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            # evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
            # Original
            # evaluator_list.append(
            #     COCOEvaluator(dataset_name, ("bbox", "segm", "keypoints"), True, output_dir=output_folder))
            evaluator_list.append(
                HsenetCOCOEvaluator(
                    dataset_name,
                    ("bbox", "segm", "keypoints"),
                    True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #    #return build_detection_test_loader(cfg, dataset_name, mapper=None)
    #    return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        # print(cfg)
        # return build_detection_train_loader(cfg, mapper=None)
        # tmp = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        # tmptmp = iter(tmp)
        # print(tmptmp.next())
        # raise Exception('Process ended')
        # return tmp
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_posture_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
