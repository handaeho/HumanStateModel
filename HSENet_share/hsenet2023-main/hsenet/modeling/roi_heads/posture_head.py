
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

ROI_POSTURE_HEAD_REGISTRY = Registry("ROI_POSTURE_HEAD")
ROI_POSTURE_HEAD_REGISTRY.__doc__ = """
Registry for posture heads, which predicts instance postures given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def posture_rcnn_loss(pred_posture_logits, instances, vis_period=0):
    """
    Compute the posture prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_posture_logits (Tensor): A tensor of shape (B, C, Hposture, Wposture) or (B, 1, Hposture, Wposture)
            for class-specific or class-agnostic, where B is the total number of predicted postures
            in all images, C is the number of foreground classes, and Hposture, Wposture are the height
            and width of the posture predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_posture_logits. The ground-truth labels (class, box, posture,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        posture_loss (Tensor): A scalar tensor containing the loss.
    """
    gt_postures = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue

        gt_postures_per_image = instances_per_image.gt_classes.to(device=pred_posture_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=posture_side_len
        gt_postures.append(gt_postures_per_image)

    if len(gt_postures) == 0:
        return pred_posture_logits.sum() * 0

    gt_postures = cat(gt_postures, dim=0)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    posture_incorrect = pred_posture_logits.argmax(1) != gt_postures
    posture_accuracy = 1 - (posture_incorrect.sum().item() / max(posture_incorrect.numel(), 1.0))

    storage = get_event_storage()
    storage.put_scalar("posture_rcnn/accuracy", posture_accuracy)

    posture_loss = F.cross_entropy(pred_posture_logits, gt_postures, reduction="mean")
    return posture_loss


def posture_rcnn_inference(pred_posture_logits, pred_instances):
    """
    Convert pred_posture_logits to estimated foreground probability postures while also
    extracting only the postures for the predicted classes in pred_instances. For each
    predicted box, the posture of the same class is attached to the instance by adding a
    new "pred_postures" field to pred_instances.

    Args:
        pred_posture_logits (Tensor): A tensor of shape (B, C, Hposture, Wposture) or (B, 1, Hposture, Wposture)
            for class-specific or class-agnostic, where B is the total number of predicted postures
            in all images, C is the number of foreground classes, and Hposture, Wposture are the height
            and width of the posture predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_postures" field storing a posture of size (Hposture,
            Wposture) for predicted class. Note that the postures are returned as a soft (non-quantized)
            postures the resolution predicted by the network; post-processing steps, such as resizing
            the predicted postures to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    posture_probs_pred = F.softmax(pred_posture_logits, 1)

    num_boxes_per_image = [len(i) for i in pred_instances]
    posture_probs_pred = posture_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(posture_probs_pred, pred_instances):
        if prob.size(0):
            scores, pred_classes = prob.max(1)
            instances.pred_classes = pred_classes
            instances.scores = scores


class BasePostureRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
    """

    @configurable
    def __init__(self, *, vis_period=0):
        """
        NOTE: this interface is experimental.

        Args:
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"vis_period": cfg.VIS_PERIOD}

    def forward(self, x, instances: List[Instances], mask_head, keypoint_head):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        with torch.no_grad():
            mask = mask_head.layers(x)
            key = keypoint_head.layers(x)
        mask = F.interpolate(mask, x.shape[2:], mode='bilinear')
        key = F.interpolate(key, x.shape[2:], mode='bilinear')

        x = cat([x, mask, key], 1)
        x = self.layers(x)
        if self.training:
            return {"loss_posture": posture_rcnn_loss(x, instances, self.vis_period)}
        else:
            posture_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


@ROI_POSTURE_HEAD_REGISTRY.register()
class PostureRCNNConvHead(BasePostureRCNNHead):
    """
    A posture head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, fc_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of classes. 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels + 6 + 17
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("posture_conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.fc_norm_relus = []
        cur_channels = cur_channels * (input_shape.width - 2 * len(conv_dims)) * (input_shape.height - 2 * len(conv_dims))
        self.cur_channels = cur_channels
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(cur_channels, fc_dim)
            self.add_module("posture_fcn{}".format(k), fc)
            self.fc_norm_relus.append(fc)
            self.add_module("posture_relu{}".format(k), nn.ReLU())
            self.fc_norm_relus.append(fc)
            cur_channels = fc_dim

        self.predictor = Linear(cur_channels, num_classes)

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for posture prediction layer
        for fc_norm_relus in self.fc_norm_relus:
            nn.init.normal_(fc_norm_relus.weight, std=0.001)
            if fc_norm_relus.bias is not None:
                nn.init.constant_(fc_norm_relus.bias, 0)

        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_POSTURE_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_POSTURE_HEAD.NUM_CONV
        fc_dim = cfg.MODEL.ROI_POSTURE_HEAD.FC_DIM
        num_fc = cfg.MODEL.ROI_POSTURE_HEAD.NUM_FC
        ret.update(
            conv_dims=[conv_dim] * (num_conv),
            fc_dims=[fc_dim] * num_fc,
            conv_norm=cfg.MODEL.ROI_POSTURE_HEAD.NORM,
            input_shape=input_shape,
        )
        ret["num_classes"] = cfg.MODEL.ROI_POSTURE_HEAD.NUM_CLASSES
        return ret

    def layers(self, x):
        b = x.size(0)
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = x.view(b, self.cur_channels)
        for layer in self.fc_norm_relus:
            x = layer(x)
        return self.predictor(x)


def build_posture_head(cfg, input_shape):
    """
    Build a posture head defined by `cfg.MODEL.ROI_POSTURE_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_POSTURE_HEAD.NAME
    return ROI_POSTURE_HEAD_REGISTRY.get(name)(cfg, input_shape)
