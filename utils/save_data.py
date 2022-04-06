import os
import cv2


def str2bool(value):
    if value.lower() in ('true', '1','t','yes'):
        return True
    else:
        return False


def change(data):
    return (data > 0.5).float()

def save_data(idx,data):
    path = 'img'
    lab_path = os.path.join(path,'lab')
    pred_path = os.path.join(path,'pred')
    if not os.path.exists(lab_path):
        os.makedirs(lab_path)
        os.makedirs(pred_path)
    if len(data[0].shape) == 3:
        lab = change(data[0]).permute(2,1,0).cpu().numpy() * 255
    else:
        lab = change(data[0][0]).permute(2,1,0).cpu().numpy() * 255
    img = change(data[1][0]).permute(2,1,0).cpu().numpy() * 255
    # print(lab.shape)
    cv2.imwrite(os.path.join(lab_path,f'{str(idx)}.png'),lab)
    cv2.imwrite(os.path.join(pred_path,f'{str(idx)}.png'),img)
