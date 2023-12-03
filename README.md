# Code Implemetation
This document provides instructions for setting up and running the CIFAR-100 dataset implementation using the provided \texttt{main.py} script.

## Requirements
The following packages are required to run the script:
```
torch==1.12.0+cu102
torchvision==0.13.0+cu102
timm==0.9.2
continuum==1.2.4
easydict
yaml
```

## Implementation
To run the script with the default settings, use the following command:

```
python main.py --initial-classes=0 --incremental-classes=10 --gpu=0
```

--initial-classes: Number of initial classes in the dataset. <br>
--incremental-classes: Number of classes to add incrementally. <br>
--gpu: GPU id to use for training. Default is 0.<br>

## Inference
If you want to see the results, check the acc.ipynb file
