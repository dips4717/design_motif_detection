from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.SOLVER.HEAD_LR_FACTOR = 1.0

# ---------------------------------------------------------------------------- #
# Few shot setting
# ---------------------------------------------------------------------------- #
_C.INPUT.FS = CN()
_C.INPUT.FS.FEW_SHOT = False
_C.INPUT.FS.SUPPORT_WAY = 2
_C.INPUT.FS.SUPPORT_SHOT = 10
_C.INPUT.AUGMENTATIONS = CN()
_C.INPUT.AUGMENTATIONS.TYPE = 'Default' # 'More'
_C.INPUT.AUGMENTATIONS.APPLY_ON_SUPPORT = False
_C.INPUT.AUGMENTATIONS.ROTATION_UPPER = 25
_C.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH = CN()
_C.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS=False
_C.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM = 128
_C.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.TEMPERATURE = 0.1
_C.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT = 1.0
_C.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.DECAY = CN({'ENABLED': False})
_C.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.DECAY.STEPS = [8000, 16000]
_C.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.DECAY.RATE = 0.2
_C.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.IOU_THRESHOLD = 0.5
_C.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_VERSION = 'V1'
_C.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.REWEIGHT_FUNC = 'none'
