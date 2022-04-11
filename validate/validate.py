import torch
import monai
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
    inference = monai.inferers.SlidingWindowInferer(roi_size=320)
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
            if config['Data']['Dimension'] == '2':
                for i in range(inputs.shape[1]):
                    # outputs = model(inputs[:,i,:])
                    outputs = inference(inputs[:,i,:], model)
                    if config['Data']['Save']:
                        save_data(i, (labels[i,:], outputs))
                    preds.append(outputs[np.newaxis,:])
                preds = torch.cat(preds, axis=0)[:,0,:]
            elif config['Data']['Dimension'] == '3':
                preds = model(inputs)
            else:
                raise ValueError(f'Dimension is not unsupported. It must be 2 or 3')
            dice += DiceCoefficient(labels, (preds.sigmoid()>0.5).float())
    print(f'Dice is {dice/len(validate_load)}')

    save_path = os.path.join(config['Paths']['checkpoint_dir'], config['DEFAULT']['Name'], "%s_checkpoint_%04d.pt" % (config['DEFAULT']['Name'], epoch))
    # store best loss and save a model checkpoint
    torch.save({"arch": "Mono_2d", "model": model.state_dict()},save_path,)
    print("Saved checkpoint to: %s" % save_path)
    if dice/len(validate_load) > val_dice and epoch !=0 :
        save_path = os.path.join(config['Paths']['checkpoint_dir'], config['DEFAULT']['Name'], f"{config['DEFAULT']['Name']}_checkpoint_best_model.pt")
        torch.save({"acc":dice/len(validate_load),"arch": "Mono_2d", "model": model.state_dict()},save_path,)
        return dice/len(validate_load)
    else:
        return val_dice