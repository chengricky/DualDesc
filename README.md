# Hierarchical Visual Localization Using Multi-modal Images 

The code achieves the network with global descriptor (Net-A-VLAD) and dense local descriptor. 

If you use this code, please refer to the paper:
```
Hierarchical Visual Localization Using Multi-modal Images 
```


In this code, the backbones (Wide ResNet-18, MobileNet V2 and ShuffleNet V2) are supported in the unified network. 

The usage of the training and testing code.

`AttentionRetrieval.py` trains the NetVLAD network (with attention module). The affiliated files used in this script is
+ `arguments.py`: read the arguments of the command
+ `DataSet/loadDataset.py`: get DataSet & DataLoader of the designated train, validation or test dataset
+ `NetAVLAD/model.py`: return Net-A-VLAD network
+ `loadCkpt.py`: load checkpoint to the model
+ `DimReduction.py`: train the dimension reduction layer (PCA with whitening) based on NetVLAD. 

`Place365/train_PlacesCNN.py` trains the scene classification network.

`ScenePlaceRecognitionMain.py` tests the unified network on different datasets, meanwhile saves the results of scene classification and scene descriptors.
