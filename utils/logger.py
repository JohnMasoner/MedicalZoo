import torch
import numpy as np
import random

from visdom import Visdom

def tensor2img(x: torch.Tensor) -> np.ndarray:
    image = x[0].cpu().float().numpy()
    image = (image - image.min())/(image.max() - image.min()) *255 if image.max() > 0 else image
    image = image if len(image.shape) == 3 else image[:,random.randint(0,32-1),:]
    assert len(image.shape) == 3, "Image should have 3 dimensions"
    return image.astype(np.uint8)

class Logger(object):
    def __init__(
        self,
        n_epochs:int,
        batch_epochs:int,
        port:int=8889,
        env:str='main'
    ):
        self.viz = Visdom(port=port, use_incoming_socket=False, env=env)
        self.n_epochs = n_epochs
        self.batch_epochs = batch_epochs
        self.batch = 1
        self.epoch = 1
        self.loss_window = {}
        self.losses = {}
        self.images_window = {}


    def log(self, losses: dict=None, images: dict=None):
        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                # print(losses[loss_name].item(), type(losses[loss_name]))
                self.losses[loss_name] = losses[loss_name].item()
                # self.losses[loss_name] = losses[loss_name].data[0]
            else:
                self.losses[loss_name] += losses[loss_name].item()
                # self.losses[loss_name] += losses[loss_name].date[0]
        # losses value
        # if (self.batch % self.batch_epochs) == 0:
        if (self.batch % self.batch_epochs) == 0:
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_window:
                    self.loss_window[loss_name] = self.viz.line(X = np.array([self.epoch]), Y=np.array([loss/self.batch]), opts={'xlabel':'step', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X = np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_window[loss_name], update='append',opts={'xlabel':'step', 'ylabel': loss_name, 'title': loss_name})

                self.losses[loss_name] = 0

            self.epoch += 1
            self.batch = 1
        else:
            self.batch += 1


        # plot images
        for image_name, image in images.items():
            # print(image_name, type(image), image.shape, image.data)
            if image_name not in self.images_window:
                self.images_window[image_name] = self.viz.image(tensor2img(image.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2img(image.data), win=self.images_window[image_name],opts={'title':image_name})


class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
