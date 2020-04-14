# Hierarchical Visual Localization Using Multi-modal Images 

The code achieves the network with global descriptor (Net-A-VLAD) and dense local descriptor. 

If you use this code, please refer to the paper:

> Hierarchical Visual Localization for Visually Impaired People Using Multimodal Images 

In this code, the following backbones are supported in the Dual Desc network. 

AlexNet, Wide ResNet, VGG-16, MobileNet V2 and ShuffleNet V2

The usage of the training and testing code.

`AttentionRetrieval.py` trains the Net-A-VLAD network (with attention module). The affiliated files used in this script is
+ `arguments.py`: read the arguments of the command
+ `DataSet/loadDataset.py`: get DataSet & DataLoader of the designated train, validation or test dataset
+ `NetAVLAD/model.py`: return Net-A-VLAD network
+ `loadCkpt.py`: load checkpoint to the model
+ `DimReduction.py`: train the dimension reduction layer (PCA with whitening) based on NetVLAD. 
`GenerateDecs.py` generate and save the trained descriptors from network.

## Dependencies
Pytorch
FAISS
SciKit-Learn
H5Py
