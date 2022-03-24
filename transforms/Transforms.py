import monai
import torch
import random

class RandCropData:
    def __init__(self, roi_size:int):
        self.roi_size = roi_size

    def random_size(self, img_size):
        left = random.randint(0,img_size[1] - self.roi_size - 1)
        bottom = random.randint(0, img_size[1] - self.roi_size - 1)
        return (left, bottom)

    def crop_data(self, img, left_top_point):
        return img[:,left_top_point[0]:left_top_point[0]+self.roi_size,left_top_point[1]:left_top_point[1]+self.roi_size]

    def __call__(self, sample):
        left_top_point = self.random_size(sample['label'].shape)
        for i in list(sample.keys()):
            sample[i] = self.crop_data(sample[i], left_top_point)
        return sample
