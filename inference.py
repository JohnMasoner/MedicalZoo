import torch
import glob
import monai
import numpy as np
import SimpleITK as sitk

from net.common import network
from data.common import dataset, MultiLoader
from tqdm import tqdm
from data import Dataloader2d

inference_path =  ''
save_path = ''
img_roi_size = 256
datas = glog.glob(os.path.join(inference_path,'*'))

inference = monai.inferers.SlidingWindowInferer(roi_size=int(img_roi_size))
for data in datas:
    data_path = data
    data = sitk.GetArrayFromImage(sitk.ReadImage(data))
    data = torch.from_numpy(data).unsqueeze(0).type(torch.FloatTensor).cuda(non_blocking=True)
    data = data.unsqueeze(2)
    preds = []
    for i in range(data.shape[1]):
        outputs = inference(inputs[:,i,:], model)
        outputs = outputs.sigmoid()>0.5
        outputs = torch.argmax(outputs, 1).unsqueeze(1).type(torch.FloatTensor)
        preds.append(outputs[np.newaxis,:])
    preds = torch.cat(preds, axis=0)[:,0,:]
    preds = preds.cpu().detach().numpy()
    preds = sitk.GetImageFromArray(preds)
    data_name = data_path.basename()
    sitk.WriteImage(preds, os.path.join(save_path, data_name))
