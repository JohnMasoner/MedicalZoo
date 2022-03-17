import torch
import numpy as np

from visdom import Visdom

def tensor2img(x: torch.Tensor) -> np.ndarray:
    image = x[0].cpu().float().numpy()
    image = (image - image.min())/(image.max() - image.min()) *255 if image.max() > 0 else image
    assert len(image.shape) == 3, "Image should have 3 dimensions"
    return image.astype(np.uint8)

class Logger(object):
    def __init__(
        self,
        n_epochs:int,
        batch_epochs:int,
        port:int=8889
    ):
        self.viz = Visdom(port=port, use_incoming_socket=False)
        self.n_epochs = n_epochs
        self.batch_epochs = batch_epochs
        self.batch = 0
        # self.epoch = 1
        self.loss_window = {}
        self.losses = {}
        self.images_window = {}


    def log(self, losses: dict=None, images: dict=None):
        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                # print(losses[loss_name].item(), type(losses[loss_name]))
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

        # losses value
        if (self.batch % 50) == 0:
            for loss_name, loss in losses.items():
                loss = loss.item()
                if loss_name not in self.loss_window:
                    self.loss_window[loss_name] = self.viz.line(X = np.array([self.batch]), Y=np.array([loss]), opts={'xlabel':'step', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X = np.array([self.batch]), Y=np.array([loss]), win=self.loss_window[loss_name], update='append')
                self.losses[loss_name] = 0.0

            # self.epoch += 1
            self.batch += 1
        else:
            self.batch += 1


        # plot images
        for image_name, image in images.items():
            # print(image_name, type(image), image.shape, image.data)
            if image_name not in self.images_window:
                self.images_window[image_name] = self.viz.image(tensor2img(image.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2img(image.data), win=self.images_window[image_name],opts={'title':image_name})
