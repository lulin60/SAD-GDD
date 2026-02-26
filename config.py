import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
# Base config files
_C.BASE = ['']
_C.NUM_WORKERS = 8
_C.PIN_MEMORY = True
_C.IMG_SIZE = 224
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.ENCODER = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'

# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1



# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = ''
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
# Normalization layers in SwinTransformerBlock before MLP, default: 'ln', choice: ['ln', 'bn']
_C.MODEL.SWIN.NORM_BEFORE_MLP = 'ln'
_C.MODEL.SWIN.ONLINE_DROP_PATH_RATE = 0.2
#Flow model
_C.MODEL.FLOW = CN()
_C.MODEL.FLOW.COUPLING_LAYERS = 8
_C.MODEL.FLOW.POS_EMBED_DIM = 128
_C.MODEL.FLOW.CLAMP_ALPHA = 1.9
_C.MODEL.FLOW.FLOW_ARCH = 'conditional_flow_model'
_C.MODEL.FLOW.Focal_WEIGHTING = True
_C.MODEL.FLOW.POS_BETA = 0.01
_C.MODEL.FLOW.MARGIN_TAU = 0.1
_C.MODEL.FLOW.NORMALIZER = 10
#_C.MODEL.FLOW.BGSPP_LAMBDA = 1


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 300


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    #print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    #  anomal flag 
    if args.ad.lower() == "true":
        config.AD  = True
    elif args.ad.lower() == 'false':
        config.AD = False

    # model params 
    config.MODEL.FLOW.COUPLING_LAYERS =  args.coupling_layers 
    config.AD_DIM  = args.ad_dim
    config.FEATS_L = args.feats_l

    #train params
    config.CE_ALPHA  = args.ce_alpha if hasattr(args, 'ce_alpha') else ''
    config.CONTRAST_ALPHA  = args.contrast_alpha if hasattr(args, 'contrast_alpha') else ''
    config.BGSPP_LAMBDA = args.bgspp_lambda if hasattr(args, 'bgspp_lambda') else ''

    config.MODEL_NAME =  args.model_name if hasattr(args, 'model_name') else ''
    config.SAVE_NAME =  args.save_name if hasattr(args, 'save_name') else ''
    config.AD_ALPHA = args.ad_alpha if hasattr(args, 'ad_alpha') else ''

    config.BATCH_SIZE_TRAIN = args.batch_size_train if hasattr(args, 'batch_size_train') else ''
    config.OUTPUT = args.output if hasattr(args, 'output') else ''
    config.LOCAL_RANK = args.local_rank if hasattr(args, 'local_rank') else 0

    # test params
    config.BATCH_SIZE_TEST = args.batch_size_test

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()

    config.weight_name = ''

    update_config(config, args)

    return config
