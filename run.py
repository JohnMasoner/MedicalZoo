from step import train, cross_model
import sys
import configparser

config = configparser.ConfigParser()
config.read(sys.argv[1])
if __name__ == '__main__':
    import torch
    with torch.autograd.set_detect_anomaly(True):
        if sys.argv[2] == '1':
            train.trainer(config)
        elif sys.argv[2] == '2':
            cross_model.trainer(config)
        else:
            raise ValueError('Invalid Training Type')
