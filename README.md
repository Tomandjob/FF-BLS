# Introduction：
Fusion features-based broad learning system (FF-BLS) is a classification network with fast and incremental learning. It contains a convolution module and broad learning. 

# Requirements:
Keras 2  
Tensorflow 1.15(<2.0)  
Python 3  
Opencv 3  

# Code structure：
BroadLearningSystem.py -the broad learning system  
FF-BLStrain.y -main script to extract the convolution feature, start training and testing  
Slidewindow-detect.py -Using sliding window and trained FF-BLS model to identify object  
BLS-model/ -trained models are saved here  

# How to run:
python FF-BLStrain.py #Please modify the dataset load path and model storage path.

# Paper Link:
Under review

# Author：
Yang Zhang and Ka-Veng Yuen  
If there is any question, welcome to discuss here or contact yangzhang@um.edu.mo


