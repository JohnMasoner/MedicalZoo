import torch
from torch import nn


class DiceLoss(nn.Module):  # pylint: disable=missing-docstring
    def __init__(self, dims=(2, 3, 4)):  # pylint: disable=missing-docstring
        super().__init__()
        self.dims = dims

    def forward(self, predict, gt, is_average=True):
        """
        This function calculates the dice loss between predicted and ground truth segmentation masks.

        Args:
          predict: The predicted output of a segmentation model, which is a tensor of shape (batch_size,
        num_classes, height, width).
          gt: gt stands for ground truth and refers to the true labels or annotations for the target object
        or region in an image or volume. In this context, it is used to calculate the intersection and union
        with the predicted segmentation mask to compute the Dice loss.
          is_average: A boolean parameter that determines whether to return the average dice loss across all
        batches or the sum of dice losses across all batches. Defaults to True

        Returns:
          the dice loss, which is a measure of the dissimilarity between the predicted and ground truth
        segmentation masks. The dice loss is calculated as 1 minus the dice coefficient, which is a measure
        of the overlap between the predicted and ground truth masks. The dice loss is averaged across all
        dimensions if `is_average` is True, otherwise it is summed.
        """
        intersection = torch.sum(predict * gt, dim=self.dims)
        union = torch.sum(predict * predict, dim=self.dims) + torch.sum(
            gt * gt, dim=self.dims
        )
        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
        dice_loss = 1 - dice.mean(1)
        dice_loss = dice_loss.mean() if is_average else dice_loss.sum()

        return dice_loss
