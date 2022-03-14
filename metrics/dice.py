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
