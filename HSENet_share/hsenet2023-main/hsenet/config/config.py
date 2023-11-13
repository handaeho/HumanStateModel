from detectron2.config import CfgNode as CN


def add_posture_config(cfg: CN):
    """
    Add config for posture head.
    """
    _C = cfg

    _C.MODEL.POSTURE_ON = False

    _C.MODEL.ROI_POSTURE_HEAD = CN()
    _C.MODEL.ROI_POSTURE_HEAD.NAME = "PostureRCNNConvHead"
    _C.MODEL.ROI_POSTURE_HEAD.NUM_CLASSES = 6
    _C.MODEL.ROI_POSTURE_HEAD.NUM_STACKED_CONVS = 8
    # Encoder
    _C.MODEL.ROI_POSTURE_HEAD.CONV_DIM = 256
    _C.MODEL.ROI_POSTURE_HEAD.NUM_CONV = 6
    _C.MODEL.ROI_POSTURE_HEAD.NORM = ""
    # fc layer
    _C.MODEL.ROI_POSTURE_HEAD.FC_DIM = 1024
    _C.MODEL.ROI_POSTURE_HEAD.NUM_FC = 1
    _C.MODEL.ROI_POSTURE_HEAD.NORM = ""

    _C.MODEL.ROI_POSTURE_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_POSTURE_HEAD.POOLER_RESOLUTION = 14
    _C.MODEL.ROI_POSTURE_HEAD.POOLER_SAMPLING_RATIO = 2
    #_C.MODEL.ROI_POSTURE_HEAD.NUM_COARSE_SEGM_CHANNELS = 2  # 15 or 2


# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #

    _C.MODEL.VOVNET = CN()

    _C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
    _C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.VOVNET.NORM = "FrozenBN"
    _C.MODEL.VOVNET.OUT_CHANNELS = 256
    _C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256
