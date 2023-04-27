import torch.nn.functional as F
from torch import nn


class BCELoss(nn.Module):
    """Binary Cross-Entropy Loss"""

    def forward(self, predict, gt, is_average=True):
        """
        This function calculates the binary cross-entropy loss between predicted and ground truth
        values.

        Args:
          predict: A tensor containing the predicted values.
          gt: gt stands for ground truth and refers to the true labels or values of the target variable
        in a supervised learning problem. In this case, it is a tensor containing the true binary values
        for the target variable.
          is_average: is a boolean parameter that determines whether to return the average binary
        cross-entropy loss or the sum of the losses across all samples in the batch. If is_average is
        True, the function returns the average loss, otherwise, it returns the sum of the losses.
        Defaults to True

        Returns:
          the binary cross-entropy loss between the predicted and ground truth values. The loss is
        calculated using the binary_cross_entropy() function from the PyTorch library. The size_average
        parameter is set to True by default, which means that the loss is averaged over all elements in
        the input tensors. If is_average is set to False, the loss is multiplied by the length of the
        input
        """
        predict = predict.float()
        gt = gt.float()
        bce_loss = F.binary_cross_entropy(predict, gt, size_average=True)
        bce_loss = bce_loss * len(predict) if not is_average else bce_loss

        return bce_loss
