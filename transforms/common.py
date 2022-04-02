import re
import inspect

from transforms import Transforms
from transforms.Transforms import *

def find_class_module(module)->list:
    '''
    To Find All Class Module,
    Args:
        module(module): the function Module
    Returns:
        class_name(list)
    '''
    class_name = []
    for name, _ in inspect.getmembers(module, inspect.isclass):
        class_name.append(name)
    return class_name

def append_class_module(transform_modules, config):
    transform_list = []
    for i in transform_modules:
        if 'crop' in i.lower():
            crop_size = config['Data']['CropSize']
            i = i + '(' + crop_size + ')'
        transform_list.append(eval(i))
    return transform_list

def lower_list(module_list):
    return [i.lower() for i in module_list]

def transform(config):
    '''
    the common moudule to transform the data
    '''
    transform_type = config['Data']['Transforms']
    # print(transform_type, type(transform_type), len(transform_type))
    if transform_type is None or len(transform_type) == 0:
        return None
    else:
        if transform_type.lower() == 'none':
            return None
        else:
            # To parase the config transform, using '!@#$%^&*' to split the different funcitons, Return a list.
            transform_type = [transform_type] if len(list(re.sub('[!@#$%^&*]', '', transform_type).split(','))) == 1 else list(re.sub('[!@#$%^&*]', '', transform_type).split(','))

            include_transforms = find_class_module(Transforms)
            assert False not in [i.lower() in lower_list(include_transforms) for i in transform_type], 'Unsupported transform type exists'
            return monai.transforms.Compose(append_class_module(include_transforms, config))
