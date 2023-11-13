# Copyright ETRI
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess

@META_ARCH_REGISTRY.register()
class HSENetRCNN(GeneralizedRCNN):
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        images, batched_inputs = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            #detected_instances = [x.to(self.device) for x in detected_instances]
            detected_instances = [x.cuda() for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return HSENetRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results


    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


    #def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
    def preprocess_image(self, batched_inputs: torch.Tensor):
        """
        Normalize, pad and batch the input images.
        """
        #images = [x["image"].to(self.device) for x in batched_inputs]
        #images = [x["image"].cuda() for x in batched_inputs]
        #images = [(x - self.pixel_mean) / self.pixel_std for x in images
        #batched_inputs = batched_inputs.cuda()
        print(f"\n batched_inputs.cuda().shape => {batched_inputs.cuda().shape}, batched_inputs.cuda().ndim => {batched_inputs.cuda().ndim}, batched_inputs.cuda().dtype => {batched_inputs.cuda().dtype}")
        images = (batched_inputs.cuda() - self.pixel_mean) / self.pixel_std

        images = [x.squeeze(0) for x in images.split(1)]
        print(f"\n images.len => {len(images)}")
        print(f"\n images => {images}")

        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        batched_inputs = [{'image': x} for x in batched_inputs]
        return images, batched_inputs
