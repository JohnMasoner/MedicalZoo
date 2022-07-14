import torch

def optimizer_common(config, model):
    opt = config['Optimizer']['Optimizer'].lower()
    if opt == 'adam':
        return torch.optim.Adam(model.parameters(), lr=float(config['Optimizer']['LearningRate']))
    elif opt == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=float(config['Optimizer']['LearningRate']))
    else:
        raise ValueError('Unknown optimizer')
