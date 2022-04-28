import monai
import torch
import configparser
import sys
import os
import cv2
import numpy as np

from net.common import network
from optimizer.common import optimizer_common
from data.common import dataset, MultiLoader
from losses.common import Losses
from tqdm import tqdm
from data import Dataloader2d
from utils.logger import Logger
from utils.save_data import *
from metrics.dice import Compute_MeanDice, DiceCoefficient
from validate.validate import validate


def trainer(config):
    model = network(config); val_dice = 0.0
    if config['Model']['LoadModel']:
        checkpoint = torch.load(config['Model']['LoadModel'])
        model.load_state_dict(checkpoint["model"])
        acc = 0 if 'acc' not in  checkpoint else checkpoint['acc']
        print(f'Loading checkpoint')
        val_dice = validate(config, model, 0, acc)

    # losses
    criterion = Losses(config)   # criterion loss funciton

    # optimizer
    optimizer, lr_scheduler = optimizer_common(config, model)

    # dataset loader
    train_dataload = dataset(config, 'train')

    logger = Logger(int(config['Training']['MaxEpoch']), len(train_dataload), env=config['DEFAULT']['Name'])
    for epoch in range(int(config['Training']['MaxEpoch'])):
        print("Epoch {}/{}".format(epoch+1, config['Training']['MaxEpoch']))
        print("-" * 10)
        # iterate all data
        model.train()
        for idx, data in enumerate(tqdm(train_dataload, desc="training")):
            if len(data.keys()) > 2:
                inputs = MultiLoader(data.keys(), data)
            else:
                inputs = data["image"].type(torch.FloatTensor).cuda(non_blocking=True)
            labels = data["label"].type(torch.LongTensor).cuda(non_blocking=True)
            # print(labels.shape)
            labels = torch.nn.functional.one_hot(labels, num_classes= int(config['DEFAULT']['NumClasses'])).transpose(1,-1)[...,-1].type(torch.FloatTensor).cuda(non_blocking=True)

            outputs = model(inputs)


            dice, loss = 0,0
            # dice, loss =
            if isinstance(outputs, dict):
                for i in outputs.keys():
                    total_loss = criterion(outputs[i][0], labels, outputs[i][1])
                    loss += sum(total_loss.values())
                outputs = outputs['MU'][0]
            else:
                assert (outputs.shape == labels.shape)
                total_loss = criterion(outputs, labels)
                # cal dice, make one_hot to labels embeding
                outputs = torch.argmax((outputs.sigmoid()>0.5).float(), 1).unsqueeze(1).type(torch.FloatTensor)
                labels = torch.argmax(labels, 1).unsqueeze(1).type(torch.FloatTensor)
                dice, loss = DiceCoefficient(outputs, labels), sum(total_loss.values())
                total_loss['loss'],total_loss['dice'] = loss, dice



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logger
            logger.log(
                losses = total_loss,
                images = {config['Data']['DataType']:inputs, 'labels':labels, 'preds': outputs}
            )
        lr_scheduler.step()
        if epoch % 2 == 0:
            val_dice = validate(config, model, epoch, val_dice)

if __name__ == "__main__":
    trainer(config)
