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

# Author：
Yang Zhang and Ka-Veng Yuen  
If there is any question, welcome to discuss here or contact yangzhangdr@126.com

# Cite This Project：
If you use this project in your research, please cite the following paper：

Yang Zhang, & Ka-Veng Yuen* (2021). Crack detection using fusion features-based broad learning system and image processing. *Computer-Aided Civil and Infrastructure Engineering*, 36(12), 1568-1584.

