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

def data_processor(data, min_value, max_value):
    data = np.clip(data, min_value, max_value)
    data = (data - np.mean(data))/np.std(data)
    return data

inference_path =  "*"
save_path = "save"
os.makedirs(save_path, exist_ok=True)
img_roi_size = 'a'
datas = glob.glob(os.path.join(inference_path,'T2*'))

config = configparser.ConfigParser()
config.read(sys.argv[1])
model = network(config)
checkpoint = torch.load(config['Model']['LoadModel'])
model.load_state_dict(checkpoint["model"])

inference = monai.inferers.SlidingWindowInferer(roi_size=int(img_roi_size)) if img_roi_size.isdigit() else None
for data in tqdm(datas):
    data_path = data.split('/')[-1]
    # case name eg,, 333333
    case_name = os.path.dirname(data).split('/')[-1]
    print(case_name)
    os.makedirs(os.path.join(save_path, case_name), exist_ok=True)
    ori_data = sitk.ReadImage(data)
    ori_spacing = ori_data.GetSpacing()
    ori_origin = ori_data.GetOrigin()
    ori_direction = ori_data.GetDirection()
    data = sitk.GetArrayFromImage(ori_data).astype(float)
    data = torch.from_numpy(data).unsqueeze(0).type(torch.FloatTensor).cuda(non_blocking=True)
    data = data.unsqueeze(2)
    preds = []
    for i in range(data.shape[1]):
        if inference is not None:
            outputs = inference(data[:,i,:], model)
        else:
            outputs = model(data[:,i,:])
        outputs = (outputs.sigmoid()>0.5).float()
        # outputs = torch.argmax(outputs, 1).unsqueeze(1).type(torch.FloatTensor)
        preds.append(outputs[np.newaxis,:])
    preds = torch.cat(preds, axis=0)[:,0,:]
    preds = preds.cpu().detach().numpy()[:,0,:]
    preds = sitk.GetImageFromArray(preds, isVector=False)
    preds.SetSpacing(ori_spacing)
    preds.SetDirection(ori_direction)
    preds.SetOrigin(ori_origin)
    sitk.WriteImage(preds, os.path.join(save_path, case_name, 'label.nii.gz'))
    sitk.WriteImage(ori_data, os.path.join(save_path, case_name, 'data.nii.gz'))
