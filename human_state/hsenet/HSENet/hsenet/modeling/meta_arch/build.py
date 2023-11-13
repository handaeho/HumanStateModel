# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright ETRI.
import torch

#from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn as nn


def _log_api_usage(identifier: str):
    """
    Internal function used to log the usage of different detectron2 components
    inside facebook's infra.
    """
    torch._C._log_api_usage_once("detectron2." + identifier)


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    #model.to(torch.device(cfg.MODEL.DEVICE))
    model.backbone.lateral_convs = nn.ModuleList(model.backbone.lateral_convs)
    model.backbone.output_convs = nn.ModuleList(model.backbone.output_convs)

    #model = model.cuda()
    _log_api_usage("hsenetv2.modeling.meta_arch." + meta_arch)
    return model
