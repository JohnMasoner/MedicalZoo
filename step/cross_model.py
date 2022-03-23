import monai
import torch
import configparser
import sys
import os
import numpy as np

from new_net import cross_teach, cross_stu
from new_net.common import network
from optimizer.common import optimizer_common
from data.common import dataset, MultiLoader
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
                inputs = data["image"].type(torch.FloatTensor).cuda(non_blocking=True)[0]
            labels = data["label"].type(torch.FloatTensor).cuda(non_blocking=True)[0]
            preds = []
            # for i in inputs[:,:1,:]:
                # x0, x1, x2, x3, x4, u4, u3, u2, u1, outputs = model(i.unsqueeze(0))
                # preds.append(outputs[np.newaxis,:])
            for i in range(inputs.shape[1]):
                x0, x1, x2, x3, x4, u4, u3, u2, u1, outputs = model(inputs[:,i,:1])
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
        torch.save({"arch": "Mono_2d", "model": model.state_dict()},save_path,)
        return dice/len(validate_load)
    else:
        return val_dice

def trainer(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['DEFAULT']['GPU']
    os.makedirs("{}/{}".format(config['Paths']['checkpoint_dir'], config['DEFAULT']['name']), exist_ok=True)

    model_teach, model_stu = network(config)
    model_teach = model_teach.train()
    model_stu   = model_stu.train()
    if config['Model']['LoadModel']:
        checkpoint = torch.load(config['Model']['LoadModel'])
        model_stu.load_state_dict(checkpoint["model"])
        print(f'Loading checkpoint')
        val_dice = validate(config, model_stu, 0, 0)
    val_dice = 0.0

    criterion_dice = monai.losses.DiceLoss()
    criterion_bce = torch.nn.BCELoss()
    criterion_huber = torch.nn.SmoothL1Loss()
    dice_loss_val = 0
    bce_loss_val = 0
    huber_loss_val = 0
    loss_val = 0

    # optimizer
    optimizer_teach, lr_scheduler_teach = optimizer_common(config, model_teach)
    optimizer_stu, lr_scheduler_stu = optimizer_common(config, model_stu)

    # dataset loader
    train_dataload = dataset(config, 'train')
    logger_step = 1
    logger = Logger(int(config['Training']['MaxEpoch']), len(train_dataload), env= config['DEFAULT']['name'])
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

            x0, x1, x2, x3, x4, u4, u3, u2, u1, logits = model_stu(inputs[:,:1,:])
            u4_, u3_, u2_, u1_, logits_ = model_teach(inputs[:,1:,:], x0, x1, x2, x3)

            ###################
            # teach model losss
            ###################
            dice_loss_teach = criterion_dice(logits_.sigmoid(), labels)
            bce_loss_teach = criterion_bce(logits_.sigmoid(), labels)
            huber_loss = (criterion_huber(u4,u4_) + criterion_huber(u3,u3_) + criterion_huber(u2,u2_) + criterion_huber(u1,u1_))
            loss_teach = dice_loss_teach + bce_loss_teach + huber_loss
            optimizer_teach.zero_grad()
            loss_teach.backward(retain_graph=True)
            optimizer_teach.step()

            ###################
            # stu model losssss
            ###################
            u4, u4_ = u4.detach(), u4_.detach()
            u3, u3_ = u3.detach(), u3_.detach()
            u2, u2_ = u2.detach(), u2_.detach()
            u1, u1_ = u1.detach(), u1_.detach()
            dice_loss_stu = criterion_dice(logits.sigmoid(), labels)
            bce_loss_stu = criterion_bce(logits.sigmoid(), labels)
            huber_loss = (criterion_huber(u4,u4_) + criterion_huber(u3,u3_) + criterion_huber(u2,u2_) + criterion_huber(u1,u1_))
            loss_stu = dice_loss_stu + bce_loss_stu + huber_loss
            dice = DiceCoefficient((logits.sigmoid()>0.5).float(), labels)
            optimizer_stu.zero_grad()
            loss_stu.backward()
            optimizer_stu.step()
            # optimizer_teach.step()
            # for Recoder
            # dice_loss_val += dice_loss
            # bce_loss_val + bce_loss
            # loss_val += loss

            # logger
            logger.log(
                losses = {'dice_teach':dice_loss_teach, 'bce_teach':bce_loss_teach, 'huber_loss':huber_loss, 'dice_stu':dice_loss_stu, 'bce_stu':bce_loss_stu,'dice':dice},
                images = {config['Data']['DataType']:inputs, 'labels':labels, 'preds': (logits.sigmoid()>0.5).float()}
            )
            logger_step += 1
        lr_scheduler_teach.step()
        lr_scheduler_stu.step()
        if epoch % 2 == 0:
            val_dice = validate(config, model_stu, epoch, val_dice)

if __name__ == "__main__":
    trainer(config)
