import monai

def network(config):
    net = config['Model']['Model'].lower()
    if net == 'unet':
        return monai.networks.nets.BasicUNet(spatial_dims=2, in_channels=int(config['Model']['in_channels']), out_channels=int(config['Model']['out_channels'])).cuda()
    elif net == 'trans':
        pass
    elif net == 'att_unet':
        from .unet_sptial import BasicUNet
        return BasicUNet(spatial_dims=2, in_channels = int(config['Model']['in_channels']), out_channels= int(config['Model']['out_channels'])).cuda()
    elif net == 'reg_mr':
        from ..new_net.VoxelModel.model import Reg_Model
        return Reg_Model(STN_size=(1000,1000)).cuda()
    elif net == 'cross_teach':
        import sys
        sys.path.append('..')
        from new_net import cross_teach,cross_stu
        (teach, stu) = cross_teach().cuda(), cross_stu().cuda()
    else:
        raise Exception('Unknown network type')
