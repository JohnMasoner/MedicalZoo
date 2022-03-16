import monai
import torch

def dataset(config, set_type):
    '''
    Choose a dataset to train or test
    Args:
        config['configparser']: The configuration
        set_type: The type of dataset to train or test
    Return:
        dataset_loader
    '''
    dim = config['Data']['Dimension']
    assert dim in ['2','3'], 'Please check you config file, make sure dimensions is 2 or 3'
    assert set_type in ['train','test'], 'Please check your dataset type, make sure set_type is train or test'

    if dim == '2':
        from data import Dataloader2d
        if set_type == 'train':
            adjacent_layer =None if config['Data']['AdjacentLayer'].lower() == 'none' or not config['Data']['AdjacentLayer'].isdigit() else int(config['Data']['AdjacentLayer'])
            train_dataset = Dataloader2d.MonoMedDataSets2D(config['Paths']['file_dir'], file_mode='NPY_train', data_type=config['Data']['DataType'], adjacent_layer=adjacent_layer)
            train_dataload = torch.utils.data.DataLoader(train_dataset, batch_size= int(config['Data']['BatchSize']), num_workers= int(config['Data']['NumWorkers']), shuffle = True)
            return train_dataload

        elif set_type == 'test':
            validate_dataset = Dataloader2d.MonoMedDataSets2DTest(config['Paths']['file_dir'],file_mode='NPY_val', data_type=config['Data']['DataType'])
            validate_load = torch.utils.data.DataLoader(validate_dataset, batch_size= 1)
            return validate_load
        else:
            raise ValueError('Error Set Type')
    else:
        raise ValueError('Error Data Dimension, Please check your config file')
