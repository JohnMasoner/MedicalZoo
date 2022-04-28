import monai
import torch
import re

class Losses(object):
    def __init__(self,config):
        '''
        Compute the losses
        Args:
            config['configparser']: The configuration parser
        Returns:
        '''
        loss_type = config['Losses']['losses']
        loss_type = loss_type if len(list(re.sub('[!@#$%^&*]', '', loss_type).split(','))) == 1 else list(re.sub('[!@#$%^&*]', '', loss_type).split(','))
        if len(loss_type)  <= 0:
            raise ValueError('loss_type must be greater than zero')
        loss_type = [i.lower().replace(' ','') for i in loss_type] if type(loss_type) != str else [loss_type]
        loss_dict = {}
        if 'dice' in loss_type:
            loss_dict['dice'] = monai.losses.DiceLoss()
        if 'bce' in loss_type:
            print(1)
            loss_dict['bce'] = torch.nn.BCELoss()
        if 'focal' in loss_type:
            loss_dict['focal'] = monai.losses.FocalLoss()
        if 'tversky' in loss_type:
            loss_dict['tversky'] = monai.losses.TverskyLoss()
        if 'contrastive' in loss_type:
            loss_dict['contrastive'] = monai.losses.ContrastiveLoss()

        self.loss_dict = loss_dict

    def __call__(self,predictions:torch.Tensor, labels:torch.Tensor, weights:int=1):
        predictions = predictions.sigmoid()
        # predictions = torch.argmax(predictions, 1).unsqueeze(1).type(torch.FloatTensor)
        # labels = torch.argmax(labels, 1).unsqueeze(1).type(torch.FloatTensor)
        total_loss_dict = {}
        for i in self.loss_dict.keys():
            total_loss_dict[i+'_loss'] = self.loss_dict[i](predictions, labels) * weights
        return total_loss_dict
