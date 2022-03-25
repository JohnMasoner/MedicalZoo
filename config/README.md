# Detailed explanation of parameters

### **```DEFAULT:```**
```BatchSize:``` The number of images to process in a batch. This is the number of images that will be processed in parallel.This value should preferably be **$2^n$**. For example, if you have a GPU with 16GB of memory, you can use a batch size of ```32```.

```NumWorkers:``` The number of workers to use for the data loader. This value should be equal to the number of CPU cores available in your machine. The default value is ```2```.

```Name:``` The name of the model to save.

```GPU:``` Specify the GPU to use. The model only supports single GPU. The default value is ```0```.

#### **```Paths:```**
```file_dir:``` The directory where the images are stored. **You have to set this parameter.**

```checkpoint_dir:``` The directory where the checkpoints are stored. The default path is ```./checkpoints```.

### **```Data:```**
```Dimension:``` The dimension of the images. The default value is ```2```, the meaning is that the images are 2D. If you want to use 3D images, you have to set this parameter to ```3```,but the model only supports 2D images. We will add support for 3D images in the future.

```DataType:``` The data type of the images. The default value is ```CT```. This value is the path name of the images. you could multi parameter like ```CT, PET, MRI```

```BatchSize:``` This value same as the ```BatchSize``` in  DEFAULT parameter.

```NumWorkers:``` This value same as the ```NumWorkers``` in  DEFAULT parameter.

```AdjacentLayer:``` The adjacent layer of the images. The default value is ```None```, the means that the adjacent layer is not used. When you set this parameter to ```1```, the adjacent layer is used, the adjacent layer is *AdjacentLayer + 1 + AdjacentLayer*.

```Transforms:``` The transforms to be applied to the images. The default value is ```None```, the means that no transforms are used. This value only supports the following transforms(We will add more transforms in the future):
```
- RandCropData: Randomly crop the images.
```

```CropSize:``` The size of the crop. The default value is ```320```, the means that the crop size is 320x320. It works only when ```Transforms``` is set to ```RandCropData```.

### **```Optimizer:```**
```Optimizer:``` The optimizer to be used. The default value is ```Adam```. This value only supports the following optimizers:
```
- Adam
- SGD
```

```LearningRate:``` The learning rate of the optimizer. The default value is ```0.0003``` The insteresting things is 3e-4 is a good value for the learning rate by Andrej Karpathy.

```Gamma:``` The gamma of the learning rate. The default value is ```0.1```.

```Step:``` The step of the learning rate. The default value is ```20```.

### **```Optimizer:```**
```Model:``` The model to be used. The default value is ```UNet```. This value only supports the following models(We will add more models in the future):
```
- UNet: The U-Net model. https://arxiv.org/abs/1505.04597
```

```in_channels:``` The number of input channels. The default value is ```1```.

```out_channels:``` The number of output channels, the class of th outputs. The default value is ```1```.

```LoadModel:``` The path of the model to be loaded. The default value is ```None```. If you set this parameter, the model will be loaded from the path.

### **```Training:```**
```MaxEpoch:``` The maximum number of epochs. The default value is ```200```.

### **```Visdom:```**
```Port:``` The port of the visdom server. The default value is ```8097```.