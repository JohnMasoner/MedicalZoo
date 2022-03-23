import torch

def optimizer_common(config, model):
    opt = config['Optimizer']['Optimizer'].lower()
    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config['Optimizer']['LearningRate']))
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config['Optimizer']['Step']), gamma=float(config['Optimizer']['Gamma']))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return optimizer, lr_scheduler
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(config['Optimizer']['LearningRate']))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config['Optimizer']['Step']), gamma=float(config['Optimizer']['Gamma']))
        return optimizer, lr_scheduler
    else:
        raise ValueError('Unknown optimizer')
