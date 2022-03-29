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
from metrics.dice import Compute_MeanDice, DiceCoefficient

def validate(config, model, epoch, val_dice):
    '''
    Validation the model
    Args:
        config[ConfigParser]: config parser
        model[Model]: model to validate
        epoch[int]: epoch number to be better save model
        val_dice[int]: dice value to saveing the best model
    '''
    dice = 0
    validate_load = dataset(config, 'test')
    model.eval()
    print('Validation----------------------------------')
    with torch.no_grad():
        validate_load = tqdm(validate_load, desc='Validation')
        for idx, data in enumerate(validate_load):
            if len(data.keys()) > 2:
                inputs = MultiLoader(data.keys(), data,'val')
            else:
                inputs = data["image"].type(torch.FloatTensor).cuda(non_blocking=True)
            labels = data["label"].type(torch.FloatTensor).cuda(non_blocking=True)[0]
            preds = []
            for i in range(inputs.shape[1]):
                outputs = model(inputs[:,i,:])
                outputs = outputs['MU'][0]
                preds.append(outputs[np.newaxis,:])
            preds = torch.cat(preds, axis=0)[:,0,:]
            dice += DiceCoefficient(labels, (preds.sigmoid()>0.5).float())
    print(f'Dice is {dice/len(validate_load)}')

    save_path = os.path.join(config['Paths']['checkpoint_dir'], config['DEFAULT']['Name'], "%s_checkpoint_%04d.pt" % (config['DEFAULT']['Name'], epoch))
    # store best loss and save a model checkpoint
    torch.save({"arch": "Mono_2d", "model": model.state_dict()},save_path,)
    print("Saved checkpoint to: %s" % save_path)
    if dice/len(validate_load) > val_dice:
        save_path = os.path.join(config['Paths']['checkpoint_dir'], config['DEFAULT']['Name'], f"{config['DEFAULT']['Name']}_checkpoint_best_model.pt")
        torch.save({"acc":dice/len(validate_load),"arch": "Mono_2d", "model": model.state_dict()},save_path,)
        return dice/len(validate_load)
    else:
        return val_dice

def trainer(config):
    model = network(config).train(); val_dice = 0.0
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
        for idx, data in enumerate(tqdm(train_dataload, desc="training")):
            if len(data.keys()) > 2:
                inputs = MultiLoader(data.keys(), data)
            else:
                inputs = data["image"].type(torch.FloatTensor).cuda(non_blocking=True)
            labels = data["label"].type(torch.FloatTensor).cuda(non_blocking=True)

            outputs = model(inputs)


            dice, loss = 0,0
            if len(outputs) > 0:
                for i in outputs.keys():
                    total_loss = criterion(outputs[i][0], labels, outputs[i][1])
                    loss += sum(total_loss.values())
                outputs = outputs['MU'][0]
            else:
                assert (outputs.shape == labels.shape)
                total_loss = criterion(outputs, labels)
                dice, loss = DiceCoefficient((outputs.sigmoid()>0.5).float(), labels), sum(total_loss.values())
                total_loss['loss'],total_loss['dice'] = loss, dice



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logger
            logger.log(
                losses = total_loss,
                images = {config['Data']['DataType']:inputs, 'labels':labels, 'preds': (outputs.sigmoid()>0.5).float()}
            )
        lr_scheduler.step()
        if epoch % 2 == 0:
            val_dice = validate(config, model, epoch, val_dice)

if __name__ == "__main__":
    trainer(config)
