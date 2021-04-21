Created by Yang Zhang

# Introduction
Fusion features-based broad learning system (FF-BLS) is a classification algorithm classification network with fast and incremental learning. It contains a convolution module and broad learning. 

# Requirements:
Keras 2
Tensorflow (<2.0)
Python 3
Opencv 3

# Code structure
BroadLearningSystem.py -the broad learning system
FF-BLStrain.y -main script to extract the convolution feature, start training and testing
Slidewindow-detect.py -Using sliding window and trained FF-BLS model to identify object.
FF-BLS model -trained models are saved here

# How to run:
python FF-BLStrain.py #Please modify the dataset load path and model storage path.

If there is any question, welcome to discuss here.


