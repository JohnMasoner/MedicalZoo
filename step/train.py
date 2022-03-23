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
from tqdm import tqdm
from data import Dataloader2d
from utils.logger import Logger
from metrics.dice import Compute_MeanDice, DiceCoefficient
def change(data):
    return (data > 0.5).float()

def save_data(path,idx,data):
    lab_path = os.path.join(path,'lab')
    pred_path = os.path.join(path,'pred')
    if not os.path.exists(lab_path):
        os.makedirs(lab_path)
        os.makedirs(pred_path)
    lab = change(data[0][0]).permute(2,1,0).cpu().numpy() * 255
    img = change(data[1][0]).permute(2,1,0).cpu().numpy() * 255
    # print(lab.shape)
    cv2.imwrite(os.path.join(lab_path,f'{str(idx)}.png'),lab)
    cv2.imwrite(os.path.join(pred_path,f'{str(idx)}.png'),img)

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
                inputs = data["image"].type(torch.FloatTensor).cuda(non_blocking=True)[0]
            labels = data["label"].type(torch.FloatTensor).cuda(non_blocking=True)[0]
            preds = []
            # for idx, i in enumerate(inputs):
                # outputs = model(i[:,idx,:])
                # preds.append(outputs[np.newaxis,:])
            for i in range(inputs.shape[1]):
                outputs = model(inputs[:,i,:])
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
    os.environ['CUDA_VISIBLE_DEVICES'] = config['DEFAULT']['GPU']
    os.makedirs("{}/{}".format(config['Paths']['checkpoint_dir'], config['DEFAULT']['name']), exist_ok=True)

    model = network(config).train()
    if config['Model']['LoadModel']:
        checkpoint = torch.load(config['Model']['LoadModel'])
        model.load_state_dict(checkpoint["model"])
        acc = 0 if 'acc' not in  checkpoint else checkpoint['acc']
        print(f'Loading checkpoint')
        val_dice = validate(config, model, 0, acc)
    val_dice = 0.0

    criterion_dice = monai.losses.DiceLoss()
    criterion_bce = torch.nn.BCELoss()
    dice_loss_val = 0
    bce_loss_val = 0
    loss_val = 0

    # optimizer
    optimizer, lr_scheduler = optimizer_common(config, model)

    # dataset loader
    train_dataload = dataset(config, 'train')

    logger = Logger(int(config['Training']['MaxEpoch']), len(train_dataload))
    for epoch in range(int(config['Training']['MaxEpoch'])):
        print("Epoch {}/{}".format(epoch+1, config['Training']['MaxEpoch']))
        print("-" * 10)
        # iterate all data
        loader = tqdm(train_dataload, desc="training")
        for idx, data in enumerate(loader):
            if len(data.keys()) > 2:
                inputs = MultiLoader(data.keys(), data)
            else:
                inputs = data["image"].type(torch.FloatTensor).cuda(non_blocking=True)
            labels = data["label"].type(torch.FloatTensor).cuda(non_blocking=True)

            outputs = model(inputs)
            assert (outputs.shape == labels.shape)
            dice_loss = criterion_dice(outputs.sigmoid(), labels)
            bce_loss = criterion_bce(outputs.sigmoid(), labels)
            loss = dice_loss + bce_loss
            dice = DiceCoefficient((outputs.sigmoid()>0.5).float(), labels)
            # for Recoder
            # dice_loss_val += dice_loss
            # bce_loss_val + bce_loss
            # loss_val += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logger
            logger.log(
                losses = {'total_loss': loss, 'dice_loss': dice_loss, 'bce_loss': bce_loss, 'dice':dice},
                images = {config['Data']['DataType']:inputs, 'labels':labels, 'preds': (outputs.sigmoid()>0.5).float()}
            )
        lr_scheduler.step()
        if epoch % 2 == 0:
            val_dice = validate(config, model, epoch, val_dice)

if __name__ == "__main__":
    trainer(config)
