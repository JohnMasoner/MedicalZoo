import monai

def network(config):
    net = config['Model']['Model'].lower()
    if net == 'unet':
        return monai.networks.nets.BasicUNet(spatial_dims=2, in_channels=int(config['Model']['in_channels']), out_channels=int(config['Model']['out_channels'])).cuda()
    elif net == 'trans':
        pass
    else:
        raise Exception('Unknown network type')
