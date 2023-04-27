import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):  # pylint: disable=missing-docstring
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicts, targets, is_average=True):
        """
        This function calculates the focal loss for binary classification tasks.

        Args:
          predicts: A tensor containing the predicted values from the model.
          targets: The ground truth labels for the batch of data being processed.
          is_average: A boolean parameter that determines whether to return the average loss per batch or
        the total loss per batch. Defaults to True

        Returns:
          the classification loss (cls_loss) which is calculated using the Focal Loss function. The cls_loss
        is multiplied by the batch size if is_average is False.
        """
        predicts = predicts.float()
        targets = targets.float()
        batch = predicts.shape[0]
        predicts = torch.clamp(predicts, 1e-4, 1.0 - 1e-4)
        alpha_factor = torch.ones(targets.shape).cuda() * self.alpha

        alpha_factor = torch.where(
            torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor
        )
        focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 - predicts, predicts)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

        bce = F.binary_cross_entropy(predicts, targets)
        cls_loss = focal_weight * bce

        cls_loss = torch.where(
            torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda()
        )
        cls_loss = cls_loss.sum() / torch.clamp(torch.sum(targets).float(), min=1.0)
        cls_loss = cls_loss * batch if not is_average else cls_loss

        return cls_loss


class FocalLossV1(nn.Module):  # pylint: disable=missing-docstring
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicts, targets, is_average=True):
        """
        This is a forward function that calculates the focal loss for a binary classification problem with
        multiple channels.

        Args:
          predicts: A tensor containing the predicted values for each class/channel.
          targets: The ground truth labels for the input data. It is a tensor of shape (batch_size,
        num_channels) where each element is either 0 or 1 indicating the absence or presence of a particular
        class in the input data.
          is_average: A boolean parameter that determines whether to return the average loss per batch or
        the total loss per batch. Defaults to True

        Returns:
          the calculated loss (all_cls_loss) after applying the focal loss function to the predicted and
        target values. The loss is calculated for each channel and then averaged over all channels. The
        final loss is multiplied by the batch size if is_average is False.
        """
        predicts = predicts.float()
        targets = targets.float()
        predicts = torch.clamp(predicts, 1e-4, 1.0 - 1e-4)

        batch = predicts.shape[0]
        num_channel = predicts.shape[1]
        all_cls_loss = 0

        for i in range(num_channel):
            target = targets[
                :,
                i,
            ]
            predict = predicts[
                :,
                i,
            ]
            alpha_factor = torch.ones(target.shape).cuda() * self.alpha[i]
            alpha_factor = torch.where(
                torch.eq(target, 1.0), alpha_factor, 1.0 - alpha_factor
            )
            focal_weight = torch.where(torch.eq(target, 1.0), 1.0 - predict, predict)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = F.binary_cross_entropy(predict, target)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(
                torch.ne(target, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda()
            )
            cls_loss = cls_loss.sum() / torch.clamp(torch.sum(target).float(), min=1.0)
            all_cls_loss += cls_loss

        all_cls_loss /= num_channel
        all_cls_loss = all_cls_loss * batch if not is_average else all_cls_loss

        return all_cls_loss
