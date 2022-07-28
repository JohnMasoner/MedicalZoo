from step import train
import sys
import configparser
import numpy as np
import torch
import os

config = configparser.ConfigParser()
config.read(sys.argv[1])

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
# torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = config['DEFAULT']['GPU']
os.makedirs("{}/{}".format(config['Paths']['checkpoint_dir'], config['DEFAULT']['name']), exist_ok=True)

if __name__ == '__main__':
    import torch
    with torch.autograd.set_detect_anomaly(True):
        if len(sys.argv) < 2:
            raise RuntimeError('Please provide at least two arguments')
        if sys.argv[2] == '1':
            train.trainer(config)
        elif sys.argv[2] == '2':
            cross_model.trainer(config)
        else:
            raise ValueError('Invalid Training Type')
