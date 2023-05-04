import os
import sys
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Base.config.config import get_cfg_defaults  # pylint: disable = wrong-import-position
from Base.dataset.data_prepare import run_prepare_data  # pylint: disable = wrong-import-position

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("./config.yaml")
    # cfg.merge_from_file()
    cfg.freeze()
    run_prepare_data(cfg)
