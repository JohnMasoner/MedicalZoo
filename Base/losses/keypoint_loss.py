import torch
from torch import nn


class JointsMSELoss(nn.Module):
    """Mean squared error loss"""

    def __init__(self, use_target_weight=False):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        """
        This function calculates the mean squared error loss between predicted and ground truth heatmaps for
        each joint in a batch of images.

        Args:
          output: The predicted heatmaps output by the model. It has a shape of (batch_size, num_joints,
        height, width).
          target: The ground truth heatmaps for the joints in the input image. These heatmaps are used to
        compute the loss during training.
          target_weight: `target_weight` is a weight assigned to each joint in the target heatmap. It is
        used to give more importance to certain joints during training. If `use_target_weight` is set to
        `True`, the `target_weight` parameter must be provided. If `use_target_weight` is `False

        Returns:
          the average loss per joint between the predicted heatmaps and the ground truth heatmaps.
        """
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        if self.use_target_weight and target_weight is None:
            target_weight = [1] * num_joints

        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[idx]),
                    heatmap_gt.mul(target_weight[idx]),
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    """Online hard key-point mining for MSE loss"""

    def __init__(self, use_target_weight=False, topk=8):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="none")
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        """
        The function calculates the average loss of the top-k highest losses for each sample in a batch
        using the Online Hard Example Mining (OHKM) algorithm.

        Args:
          loss: The input parameter `loss` is a PyTorch tensor representing the loss values for a batch of
        samples. The size of the tensor is `[batch_size, num_classes]`, where `batch_size` is the number of
        samples in the batch and `num_classes` is the number of classes in

        Returns:
          The function `ohkm` returns the average loss calculated using the Online Hard Example Mining
        (OHEM) technique.
        """
        ohkm_loss = 0.0
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            _, topk_idx = torch.topk(sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight=None):
        """
        This function calculates the loss between predicted and ground truth heatmaps for each joint in a
        batch using a specified criterion and returns the loss after applying online hard example mining.

        Args:
          output: The predicted heatmaps from the model. It is a tensor of shape (batch_size, num_joints,
        height, width).
          target: The ground truth heatmaps for the joints in the input image. These heatmaps are used to
        compute the loss between the predicted heatmaps and the ground truth heatmaps.
          target_weight: `target_weight` is a weight tensor that is used to weight the contribution of each
        joint to the overall loss. If `use_target_weight` is set to `True`, then `target_weight` is
        expected to be a tensor of shape `(batch_size, num_joints)` where each element represents

        Returns:
          the output of the `ohkm` function applied to the `loss` tensor. The `ohkm` function likely
        performs some form of online hard example mining to select the most difficult examples to train on.
        """
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        if self.use_target_weight and target_weight is None:
            target_weight = [1] * num_joints

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if batch_size == 1:
                heatmap_pred = heatmap_pred.unsqueeze(dim=0)
                heatmap_gt = heatmap_gt.unsqueeze(dim=0)
            if self.use_target_weight:
                loss.append(
                    0.5
                    * self.criterion(
                        heatmap_pred.mul(target_weight[:, idx]),
                        heatmap_gt.mul(target_weight[:, idx]),
                    )
                )
            else:
                loss.append(0.5 * self.criterion(heatmap_pred, heatmap_gt))

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


class KeypointFocalLoss(nn.Module):
    """Focal loss for key-point detection, refer to centerNet."""

    def forward(self, pred, gt):
        """
        This function calculates the forward pass of a custom loss function for binary classification tasks.

        Args:
          pred: The predicted values from the model, which are expected to be between 0 and 1.
          gt: gt stands for ground truth and is a tensor containing the true labels for a set of samples. In
        this context, it is used to calculate the positive and negative indices for the loss function.

        Returns:
          the calculated loss value.
        """
        delta = 1e-7
        pred = pred.float()
        gt = gt.float()

        pred = torch.clamp(pred, 1e-4, 1.0 - 1e-4)

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred + delta) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = (
            torch.log(1 - pred + delta) * torch.pow(pred, 2) * neg_weights * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()

        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss
