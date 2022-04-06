import monai
import torch
import random

class RandCropData:
    def __init__(self, roi_size:int):
        self.roi_size = roi_size

    def random_size(self, img_size):
        # left = random.randint(0,img_size[1] - self.roi_size - 1)
        # bottom = random.randint(0, img_size[1] - self.roi_size - 1)
        left = random.randint(0,max(img_size) - self.roi_size - 1)
        bottom = random.randint(0, max(img_size) - self.roi_size - 1)
        return (left, bottom)

    def crop_data(self, img, left_top_point, types):
        if types == 'train':
            return img[:,left_top_point[0]:left_top_point[0]+self.roi_size,left_top_point[1]:left_top_point[1]+self.roi_size]
        elif types == 'test':
            return img[:,:,left_top_point[0]:left_top_point[0]+self.roi_size,left_top_point[1]:left_top_point[1]+self.roi_size]
    def __call__(self, sample, types='train'):
        left_top_point = self.random_size(sample['label'].shape)
        for i in list(sample.keys()):
            sample[i] = self.crop_data(sample[i], left_top_point, types)
        return sample

class LabelCrop:
    def __init__(self, roi_size:int):
        self.roi_size = roi_size

    def read_label_point(self, label):
        label_data =  torch.where(label==1)
        if len(label_data[0]) == 0:
            return (0,256,256)
        rnd_point = random.randint(0, len(label_data[0]))
        return (0, label_data[1][rnd_point], label_data[2][rnd_point])

    def get_crop_point(self, data):
        roi_size = self.roi_size // 2
        left_point = data[-2] - roi_size
        if left_point <0:
            return (0, self.roi_size)
        top_point = data[-1] - roi_size
        if top_point < 0:
            return (self.roi_size, 0)

        return (left_point, top_point)

    def crop_data(self, img, left_top_point, types):
        if types == 'train':
            return img[:,left_top_point[0]:left_top_point[0]+self.roi_size,left_top_point[1]:left_top_point[1]+self.roi_size]
        elif types == 'test':
            return img[:,:,left_top_point[0]:left_top_point[0]+self.roi_size,left_top_point[1]:left_top_point[1]+self.roi_size]

    def __call__(self, sample, types='train'):
        if types =='test':
            return sample
        label_point = self.read_label_point(sample['label'])
        left_top_point = self.get_crop_point(label_point)
        for i in list(sample.keys()):
            sample[i] = self.crop_data(sample[i], left_top_point, types)
        return sample

class GenerateMask:
    def __init__(self, num_patches, prob:float=0.3):
        self.num_patches = num_patches
        self.prob = prob

    def mask(self, data):
        if float(data.max().item()) < 0:
            return data
        else:
            if random.random() <= self.prob:
                return data.quantile(0.5, keepdim=True)
            else:
                return data

    def scan_data(self, data):
        shape = data.shape[-1]
        for i in range(shape // self.num_patches):
            for j in range(shape // self.num_patches):
                data[:, i*self.num_patches:(i+1)*self.num_patches, j*self.num_patches:(j+1)*self.num_patches] = self.mask(data[:, i*self.num_patches:(i+1)*self.num_patches, j*self.num_patches:(j+1)*self.num_patches])
        return data

    def __call__(self, sample):
        for i in list(sample.keys()):
            if i == 'label':
                sample[i] = sample[i]
            else:
                sample[i] = self.scan_data(sample[i])
        return sample

