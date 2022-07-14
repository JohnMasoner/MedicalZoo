import torch
from data.common import MultiLoader, dataset
from losses.common import Losses
from metrics.dice import DiceCoefficient
from net.common import network
from optimizer.common import optimizer_common
from torch import nn
from tqdm import tqdm
from utils.logger import Logger
from validate.validate import validate


class SegmentationNetwork(nn.Module):

    def __init__(self, config, phase: 'str' = 'train'):
        self.config = config
        self.phase = phase

        self.model = network(self.config)
        self._init_optimizer()
        self._init_lr_scheduler()

        self._init_loss()
        self._load_weights()

        pass

    def _init_optimizer(self):
        self.optimizer = optimizer_common(self.config, self.model)

    def _init_loss(self):
        self.loss = Losses(self.config)

    def _init_lr_scheduler(self):
        if self.config.get('lr_scheduler', 'type') == 'step':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.getint('lr_scheduler', 'step_size'),
                gamma=self.config.getfloat('lr_scheduler', 'gamma'))
        elif self.config.get('lr_scheduler', 'type') == 'cos':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.getint('lr_scheduler', 'T_max'),
                eta_min=self.config.getfloat('lr_scheduler', 'eta_min'))
        else:
            raise NotImplementedError(
                'lr_scheduler type not implemented, please check config file \
                , only support step and cos')

    def _load_weights(self):
        if self.config.get('Model', 'LoadModel') and self.phase == 'train':
            checkpoint = torch.load(self.config.get('Model', 'LoadModel'))
            self.model.load_state_dict(checkpoint["model"])
            self.acc = 0 if 'acc' not in checkpoint else checkpoint['acc']

    def _create_train_dataloader(self):
        self.train_dataload = dataset(self.config, 'train')
        self.val_dataload = dataset(self.config, 'test')

    def do_train(self):
        self.logger.info("Start training...")
        self.logger.info("Epoch number: {}".format(
            self.config.getint('train', 'epochs')))
        self.logger.info("Batch size: {}".format(
            self.config.getint('train', 'batch_size')))

        self._create_train_datal1oader()
        self.logger = Logger(int(self.config['Training']['MaxEpoch']),
                             len(self.train_dataload),
                             env=self.config['DEFAULT']['Name'])

        for epoch in range(self.config.get('Training', 'MaxEpoch')):
            self.logger.start_epoch(epoch)
            self.lr_scheduler.step()
            self.model.train()
            self._train()
            if epoch % self.config.getint('Training', 'validate_inter') == 0:
                validate(self.config, self.model, epoch, self.dice)

    def _train(self):
        self.model.train()
        for i, data in enumerate(tqdm(self.train_dataload, desc="training")):
            if len(data.keys()) > 2:
                inputs = MultiLoader(data.keys(), data)
            else:
                inputs = data["image"].type(
                    torch.FloatTensor).cuda(non_blocking=True)

            if int(self.config['DEFAULT']['NumClasses']) > 1:
                labels = data["label"].type(
                    torch.LongTensor).cuda(non_blocking=True)
                # print(labels.shape)
                labels = torch.nn.functional.one_hot(
                    labels,
                    num_classes=int(
                        self.config['DEFAULT']['NumClasses'])).transpose(
                            1, -1)[..., -1].type(
                                torch.FloatTensor).cuda(non_blocking=True)
            else:
                labels = data["label"].type(
                    torch.FloatTensor).cuda(non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = 0
            if isinstance(outputs, dict):
                for i in outputs.keys():
                    total_loss = self.loss(outputs[i][0], labels,
                                           outputs[i][1])
                    loss += sum(total_loss.values())
                outputs = outputs['MU'][0]
            else:
                assert (outputs.shape == labels.shape)
                total_loss = self.loss(outputs, labels)
                if int(self.config['DEFAULT']['NumClasses']) > 1:
                    outputs = (outputs.sigmoid() > 0.5).float()
                    dice, loss = DiceCoefficient(
                        outputs,
                        labels,
                        num_classes=int(
                            self.config['DEFAULT']['NumClasses'])), sum(
                                total_loss.values())
                    # cal dice, make one_hot to labels embeding
                    outputs = torch.argmax(outputs, 1).unsqueeze(1).type(
                        torch.FloatTensor)
                    labels = torch.argmax(labels, 1).unsqueeze(1).type(
                        torch.FloatTensor)
                else:
                    outputs = (outputs.sigmoid() > 0.5).float()
                    dice, loss = DiceCoefficient(outputs, labels), sum(
                        total_loss.values())
                total_loss['loss'], total_loss['dice'] = loss, dice

            # the following code is for show the input and output
            if self.config['Data']['AdjacentLayer'].isdigit():
                adjacent_layer = int(self.config['Data']['AdjacentLayer'])
                labels = labels[:, adjacent_layer:adjacent_layer + 1, :]
                inputs = inputs[:, adjacent_layer:adjacent_layer + 1, :]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.dice = dice

            self.logger.log(losses=total_loss,
                            images={
                                self.config['Data']['DataType']: inputs,
                                'labels': labels,
                                'preds': outputs
                            })
