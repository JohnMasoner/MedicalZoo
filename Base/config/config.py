import os

import yaml
from yacs.config import CfgNode as CN

######################
# config definition
######################
_C = CN()

########################
# Environment definition
########################
_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.CUDA = True
_C.ENVIRONMENT.SEED = 3407
_C.ENVIRONMENT.AMP_OPT_LEVEL = "O1"
_C.ENVIRONMENT.NUM_GPU = 0
_C.ENVIRONMENT.MONITOR_TIME_INTERVAL = 0.1
_C.ENVIRONMENT.DATA_BASE_DIR = None
_C.ENVIRONMENT.PHASE = "train"
_C.ENVIRONMENT.NAME = "adam"
_C.ENVIRONMENT.EXPERIMENT_PATH = "experiment"
_C.ENVIRONMENT.FINE_MODE = True

########################
# Dataset definition
########################
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 24
# num_classes is x + 1, when single-class is 0 + 1
_C.DATA.NUM_CLASSES = 0 + 1
_C.DATA.NUM_WORKERS = 2
_C.DATA.MAIN_MODAL = ["CT"]
_C.DATA.SUB_MODAL = []
_C.DATA.LABEL = ["label"]
_C.DATA.DIMENSION = 2
_C.DATA.IN_CHANNELS = 1
_C.DATA.NER_LAYER = 0
_C.DATA.FILE_PATH_TRAIN = "/raid0/xxx/Y064/Dataset/NPY_train"
_C.DATA.FILE_PATH_VAL = "/raid0/xxx/Y064/Dataset/NPY_val"
_C.DATA.SAVE = False
_C.DATA.SAVE_FILE = "/raid0/xxx/AdaptationDataset/save/"

########################
# Model definition
########################
_C.MODEL = CN()
_C.MODEL.NAME = "crossAttention"
_C.MODEL.TYPE = "unet"
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.RESUME = ""
_C.MODEL.ZERO_TRAIN = False
_C.MODEL.TEST_MODE = False

########################
# Training definition
########################
_C.TRAINING = CN()
_C.TRAINING.START_EPOCH = 0
_C.TRAINING.EPOCH = 100
_C.TRAINING.LOSS = "diceAndBce"
_C.TRAINING.ACTIVATION = "sigmoid"
_C.TRAINING.METRIC = "dice"
_C.TRAINING.LR_SCHEDULER = "CosineLR"
_C.TRAINING.LOSS_SMOOTH = False
# OPTIMIZER config
_C.TRAINING.OPTIMIZER = CN()
_C.TRAINING.OPTIMIZER.METHOD = "adam"
_C.TRAINING.OPTIMIZER.BASE_LR = 3e-4
_C.TRAINING.OPTIMIZER.WEIGHT_DECAY = 5e-5

# Test Time Adaptation
_C.TRAINING.TTA = CN()
_C.TRAINING.TTA.MODE = "target"
_C.TRAINING.TTA.RESUME = ""
_C.TRAINING.TTA.UPDATE_PRARMS = "all-BN"
_C.TRAINING.TTA.CONSISLEARNING = False
_C.TRAINING.TTA.BLOCK_BATCH = False

# Display the config
_C.DISPLAY = CN()
_C.DISPLAY.TRAIN = True
_C.DISPLAY.TRAIN_ITER = 100
_C.DISPLAY.TRAIN_IMAGE = False
_C.DISPLAY.TRAIN_LINE = True
_C.DISPLAY.VAL = True
_C.DISPLAY.VAL_ITER = 100
_C.DISPLAY.VAL_IMAGE = True
_C.DISPLAY.VAL_LINE = True

########################
# Data Augmentation definition
########################
_C.DATA_AUG = CN()
_C.DATA_AUG.IS_ENABLE = True
_C.DATA_AUG.IS_SHUFFLE = False
_C.DATA_AUG.IS_RANDOM_FLIP = False
_C.DATA_AUG.IS_RANDOM_ROTATE = False
_C.DATA_AUG.IS_RANDOM_ZOOM = False
_C.DATA_AUG.IS_RANDOM_GAUSS_SHARP = True
_C.DATA_AUG.IS_RANDOM_GAUSS_SMOTH = True
_C.DATA_AUG.IS_RANDOM_HIS_SHIFT = True
_C.DATA_AUG.IS_SHUFFLE_REMAP = False
_C.DATA_AUG.IS_SYNC_TUMOR = False
_C.DATA_AUG.IS_RANDOM_CROP = False


def _update_config_from_file(config, cfg_file):
    """
    This function updates a configuration file with values from a YAML file and merges it with the
    existing configuration.

    Args:
      config: The `config` parameter is an instance of a configuration object that is used to store and
    manage various settings and parameters for a program or system. It is likely defined using a
    configuration library such as `yacs` or `argparse`.
      cfg_file: The `cfg_file` parameter is a string representing the path to a YAML configuration file.
    """
    config.defrost()
    with open(cfg_file, "r", encoding="utf-8") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print(f"=> merge config from {cfg_file}")
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):  # pylint: disable=missing-docstring
    _update_config_from_file(config, args.cfg)


def get_config(args):  # pylint: disable=missing-docstring
    """
    This function returns a modified configuration object based on the input arguments.

    Args:
      args: The `args` parameter is likely a command-line argument parser object that contains the
    arguments passed to the script or function. It is used to update the default configuration settings
    in `_C` by calling the `update_config` function. The updated configuration is then returned by the
    `get_config` function.

    Returns:
      The function `get_config` is returning a configuration object that is created by cloning the
    default configuration object `_C` and updating it with the arguments passed to the function.
    """
    config = _C.clone()
    update_config(config, args)

    return config
