import numpy as np
import torch
from torch import nn


class FlattenBCE(nn.BCELoss):  # pylint: disable=missing-docstring
    def __init__(self, reduce=False):
        super().__init__(reduce=reduce)

    def forward(self, predict, target):
        """
        This function flattens the input tensors and applies binary cross-entropy loss.

        Args:
          predict: A tensor representing the predicted output of a model. It is expected to be a contiguous
        tensor and is reshaped to a 1D tensor using the view() method.
          target: The target parameter is a tensor containing the ground truth values for the binary
        classification task. It should have the same shape as the predict tensor.

        Returns:
          The `forward` method of the `FlattenBCE` class is being overridden to take in `predict` and
        `target` as inputs, flatten them using the `view` method, and then pass them to the `forward` method
        of the parent class (`nn.BCELoss`). The output of the parent class's `forward` method is then
        returned. Therefore, the method returns the
        """
        predict = predict.contiguous()
        predict = predict.view(
            -1,
        )
        target = target.contiguous()
        target = target.view(
            -1,
        )

        return super().forward(predict, target)


class TopKLoss(nn.Module):  # pylint: disable=missing-docstring
    def __init__(self, k=(10,)):
        super().__init__()
        self.k = k

    def forward(self, predict, target, is_average=True):
        """
        This function calculates the forward pass of a neural network using binary cross-entropy loss and
        selects the top k percent of voxels with the highest loss for each channel.

        Args:
          predict: A tensor containing the predicted output from a neural network. It has shape (batch_size,
        num_channels, height, width, depth).
          target: The ground truth target values for the prediction.
          is_average: A boolean parameter that determines whether to return the average loss per batch or
        the total loss per batch. Defaults to True

        Returns:
          the total loss calculated using the FlattenBCE loss function for each channel in the predicted
        output and the target output. The loss is then averaged across all channels and multiplied by the
        batch size if `is_average` is False.
        """
        all_loss = 0
        batch = predict.shape[0]
        num_channel = predict.shape[1]
        assert len(self.k) == num_channel
        for i in range(num_channel):
            flatten_bce_loss = FlattenBCE()
            loss = flatten_bce_loss(predict[:, i], target[:, i])
            num_voxels = np.prod(loss.shape)
            loss, _ = torch.topk(
                loss.view((-1,)), int(num_voxels * self.k[i] / 100), sorted=False
            )
            all_loss += loss.mean()
        all_loss /= num_channel
        all_loss = all_loss * batch if not is_average else all_loss

        return all_loss
