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
    elif net == 'multi_loss':
        import sys
        sys.path.append('..')
        from new_net import ModelFuse
        return ModelFuse.Module(multi_channels=None).cuda()
    elif net == 'unetr':
        return monai.networks.nets.UNETR(in_channels=int(config['Model']['in_channels']), out_channels=int(config['Model']['out_channels']), img_size=(320,320), spatial_dims=2).cuda()
    elif net == 'vit':
        return monai.networks.nets.ViT(in_channels=int(config['Model']['in_channels']), img_size= (320,320), pos_embed='conv', patch_size= 16, spatial_dims=2).cuda()
    elif net == 'vh':
        import sys
        sys.path.append('..')
        from new_net.Transformer import VHUFormer
        return VHUFormer.VHUFormer(in_chns=int(config['Model']['in_channels']), num_classes= int(config['Model']['out_channels']), num_layers=50,embed_dim=768, hidden_features=400, img_size= (320,320), patch_size= (16,16), hypbird=True).cuda()
    elif net == 'transmed':
        import sys
        sys.path.append('..')
        from new_net.Transformer import TransMed
        return TransMed.TransMed(in_chns=int(config['Model']['in_channels']), root_u_out_chns=256).cuda()
    elif net == 'transfuse':
        import sys
        sys.path.append('..')
        from new_net.Transformer import TransFuse
        return TransFuse.DownBlockTrans(4, 16, embed_dim=512).cuda()

    else:
        raise Exception('Unknown network type')
