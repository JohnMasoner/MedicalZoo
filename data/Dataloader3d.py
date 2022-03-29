import os
import numpy as np
import glob
import torch
import random
import monai

from transforms import Transforms
from data import Dataloader2d

class MedDataSets3D(Dataloader2d.MultiMedDatasets2DTest):
    def __init__(self, file_dir:str=None, file_mode = None, data_type = None):
        self.file_dir = os.path.join(file_dir, file_mode)
        self.label = sorted(glob.glob(os.path.join(self.file_dir,'*/label/')))
        self.adjacent_layer = None
        self.data_type = [i for i in data_type.replace(' ','').split(',')]

    def __getitem__(self, idx):
        sample = {}
        sample['label'] = self.Read3DData(self.label[idx])
        for i in self.data_type:
            sample[f'{i}'] = self.Read3DData(self.label[idx].replace('label', i))
        sample = self.RandCropLayer(sample)

        return sample

    def Read3DData(self, modal_path):
        data_list = [os.path.join(modal_path,f'{str(i)}.npy') for i in range(len(glob.glob(os.path.join(modal_path,'*'))))]
        data = []
        for idx, i in enumerate(data_list):
            data.append(self.ReadData(i)[np.newaxis,:])
        data = self.Normalization(np.vstack(data), 4)
        return data

    def RandCropLayer(self, sample):
        d_shape = sample['label'].shape[1]
        start_layer = random.randint(0,d_shape-32-1)
        for i in sample.keys():
            sample[i] = sample[i][:,start_layer:start_layer+32,:]
        return sample
