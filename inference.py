import torch
import configparser
import os
import sys
import glob
import monai
import numpy as np
import SimpleITK as sitk

from net.common import network
from data.common import dataset, MultiLoader
from tqdm import tqdm
from data import Dataloader2d

inference_path =  "/raid0/myk/FLARE2022/TestImage/"
save_path = "/raid0/myk/FLARE2022/TestImage_inf/"
os.makedirs(save_path, exist_ok=True)
img_roi_size = 256
datas = glob.glob(os.path.join(inference_path,'*'))

config = configparser.ConfigParser()
config.read(sys.argv[1])
model = network(config)
checkpoint = torch.load(config['Model']['LoadModel'])
model.load_state_dict(checkpoint["model"])


inference = monai.inferers.SlidingWindowInferer(roi_size=int(img_roi_size))
for data in tqdm(datas):
    data_path = data.split('/')[-1]
    data = sitk.GetArrayFromImage(sitk.ReadImage(data))
    print(data.shape)
    data = torch.from_numpy(data).unsqueeze(0).type(torch.FloatTensor).cuda(non_blocking=True)
    data = data.unsqueeze(2)
    preds = []
    for i in range(data.shape[1]):
        outputs = inference(data[:,i,:], model)
        outputs = (outputs.sigmoid()>0.5).float()
        outputs = torch.argmax(outputs, 1).unsqueeze(1).type(torch.FloatTensor)
        preds.append(outputs[np.newaxis,:])
    preds = torch.cat(preds, axis=0)[:,0,:]
    preds = preds.cpu().detach().numpy()[:,0,:]
    print(preds.shape)
    preds = sitk.GetImageFromArray(preds)
    sitk.WriteImage(preds, os.path.join(save_path, data_path))
