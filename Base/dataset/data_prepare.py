import json
import os
import sys
import traceback
from glob import glob
from multiprocessing import Pool
from multiprocessing import cpu_count

import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(BASE_DIR)

from Common.image_io import load_nii_file  # pylint: disable = wrong-import-position
from Common.image_io import save_npy_file  # pylint: disable = wrong-import-position
from Common.image_resample import MedicalImageResampler  # pylint: disable = wrong-import-position


def run_prepare_data(cfg):
    """
    This function prepares data using multiprocessing to apply a process to each data item in the
    data_info list.

    Args:
      cfg: The variable `cfg` is likely a configuration object or dictionary that contains various
    settings and parameters for the data preparation process. It is passed as an argument to the
    `DataPrepare` class constructor and is used to configure the behavior of the data preparation
    process.
    """
    data_prepare = DataPrepare(cfg)

    pool = Pool(int(cpu_count() * 0.1))  # pylint: disable = consider-using-with
    for data in data_prepare.data_info:
        try:
            pool.apply_async(data_prepare.process, (data,))
            # data_prepare.process(data)
        except Exception as err:
            traceback.print_exc()
            print(
                f"Create coarse/fine image/mask throws exception {err}, with series_id {data.series_id}!"
            )

    pool.close()
    pool.join()

    print("Finish data prepare!")


class DataInfo:  # pylint: disable = missing-class-docstring
    def __init__(
        self, series_id, image_path, mask_path=None
    ):  # pylint: disable = missing-class-docstring
        self.series_id = series_id
        self.image_path = image_path
        self.mask_path = mask_path


class DataPrepare:  # pylint: disable = missing-class-docstring
    def __init__(self, cfg):  # pylint: disable = missing-class-docstring
        super().__init__()
        self.cfg = cfg
        self.base_type = cfg.DATA_PREPARE.BASE_TYPE
        self.base_dir = cfg.DATA_PREPARE.BASE_DIR
        self.save_dir = cfg.DATA_PREPARE.SAVE_DIR
        self.spacing = cfg.DATA_PREPARE.SPACING
        self.crop_background = cfg.DATA_PREPARE.CROP_BACKGROUND
        self.modal_name = cfg.DATA_PREPARE.MODAL_NAME
        self.label_name = cfg.DATA_PREPARE.LABEL_NAME
        self.label_num = cfg.DATA.NUM_CLASSES
        self.dim = cfg.DATA.DIMENSION

        if self.base_type == "nnUNet":
            self.image_dir = (
                os.path.join(self.base_dir, "imagesTr")
                if os.path.exists(os.path.join(self.base_dir, "imagesTr"))
                else None
            )
            self.mask_dir = (
                os.path.join(self.base_dir, "labelsTr")
                if os.path.exists(os.path.join(self.base_dir, "labelsTr"))
                else None
            )
            self.data_json = os.path.join(self.base_dir, "dataset.json")
        elif self.base_type == "Custom":
            self.image_dir = self.base_dir
            self.mask_dir = None
            self.data_json = None
        else:
            raise ValueError("Base type must be nnUNet or Custom!")

        self.data_info = self._read_info()

    def process(self, data):  # pylint: disable = missing-function-docstring
        series_id = data.series_id
        image_path = data.image_path
        mask_path = data.mask_path
        # print(image_path, mask_path)
        # for image in image_path:
        image_info = load_nii_file(image_path)
        modal_type = os.path.basename(image_path).split(".")[0]
        npy_image = image_info["npy_image"]
        # image_direction = image_info["direction"]
        image_spacing = image_info["spacing"]

        mask_info = load_nii_file(mask_path)
        mask_type = os.path.basename(mask_path).split(".")[0]
        npy_mask = mask_info["npy_image"]
        # mask_direction = mask_info["direction"]
        mask_spacing = mask_info["spacing"]

        npy_image, _, _ = MedicalImageResampler.resample_to_spacing(
            npy_image, image_spacing, self.spacing
        )
        if self.label_num > 1:
            num_label = int(self.label_num)
            npy_mask, _, _ = MedicalImageResampler.resample_overlap_mask_to_spacing(
                npy_mask, mask_spacing, self.spacing, num_label
            )
        else:
            num_label = np.max(npy_mask)
            npy_mask, _, _ = MedicalImageResampler.resample_mask_to_spacing(
                npy_mask, mask_spacing, self.spacing, num_label
            )
        if self.dim == 2:
            save_npy_file(
                npy_image,
                os.path.join(self.save_dir, series_id, modal_type),
                dim=self.dim,
            )
            save_npy_file(
                npy_mask,
                os.path.join(self.save_dir, series_id, mask_type),
                dim=self.dim,
            )
        elif self.dim == 3:
            save_npy_file(
                npy_image,
                os.path.join(self.save_dir, series_id, modal_type + ".npy"),
                dim=self.dim,
            )
            save_npy_file(
                npy_mask,
                os.path.join(self.save_dir, series_id, mask_type + ".npy"),
                dim=self.dim,
            )

        print(f"End processing {series_id}'s {modal_type}.")

    def _read_info(self):  # pylint: disable = missing-class-docstring
        local_info = []
        # TODO: read nnUNet dataset
        if self.base_type == "nnUNet":
            patient_name = [i.split(".")[0] for i in os.listdir(self.mask_dir)]
        elif self.base_type == "Custom":
            patient_name = os.listdir(self.image_dir)
            for name in patient_name:
                for modal in self.modal_name:
                    image_path = glob(os.path.join(self.base_dir, name, modal + "*"))[0]
                    mask_path = glob(
                        os.path.join(self.base_dir, name, self.label_name + "*")
                    )[0]
                    local_info.append(DataInfo(name, image_path, mask_path))

        print(
            f"There are {len(patient_name)} patients have {len(local_info)} datas in total!"
        )
        print("Finish reading data info!")
        return local_info

    def _corr_modal(self):  # pylint: disable = missing-class-docstring
        data_info = json.load(
            open(  # pylint: disable = consider-using-with
                self.data_json, "r", encoding="utf8"
            )
        )
        modal_name = data_info["channel_names"]
        return modal_name
