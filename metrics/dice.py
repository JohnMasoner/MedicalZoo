import torch

def Compute_MeanDice(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert not torch.any(y_pred > 1), 'Please check the y_pred value, you can try to sigmoid.'
    assert not torch.any(y_pred < 0), 'Please check the y_pred value, you can try to sigmoid.'

    inputs = input.float()
    target = target.float()
    noreducedim = [0] + list(range(2, len(inputs.shape)))

    intersect = (inputs * target).sum(dim=noreducedim)
    denominator = (inputs + target).sum(dim=noreducedim)
    dices = (2 * intersect + 1e-6) / (denominator + 1e-6)
    return dices.mean().item()

def DiceCoefficient(y_pred, y_true, smooth=1e-6):
    y_true_f = torch.flatten(y_true).cuda()
    y_pred_f = torch.flatten(y_pred).cuda()
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class MultiClassDiceCoefficient():
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def __call__(self,inputs, target):
        target = _one_hot_encoder(target)
        dice = 0
        for i in range(0, self.n_classes):
            dice = DiceCoefficient(inputs, target).item()
            dice += dice
        return dice
