import numpy as np
import os
import cv2
from BroadLearningSystem import BLStrain, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes, bls_train_input, \
    bls_train_inputenhance
import keras

def transformLabel(label_raw):######################################################此时类别是两个类

    #results = [0, 0]
    if label_raw == 1: ##
        results = [0, 1]
    #elif label_raw == '0':
    else:
        results = [1, 0]
    return results

def rotateImg(img_raw):
    rows, cols = img_raw.shape[:2]
    result_img = []
    for i in range(10):
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), i * 5, 1)
        dest_img = cv2.warpAffine(img_raw, M, (rows, cols))
        result_img.append(dest_img)

        M2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), -i * 5, 1)
        dest_img2 = cv2.warpAffine(img_raw, M2, (rows, cols))
        result_img.append(dest_img2)

    return result_img

#######卷积层处理#####
from keras.models import Sequential,Model,Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50


############通过txt文件进行数据读取，txt存储有图片的路径和类别
def getDatanew(filePath, resize_format=(100, 100), colorType=cv2.IMREAD_COLOR, resize_interpolation=cv2.INTER_NEAREST,
            needGenImg=False):####IMREAD_COLOR   IMREAD_GRAYSCALE
    tmpData = []
    tmpLabel = []

    ##
    model = ResNet50(include_top=True, weights='G:/zy/01/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    dense_result = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)  ####分类层的前一层

    with open(filePath) as txtData:
        lines = txtData.readlines()
        for line in lines:
            file, label = line.split(' ')
            #print(label)
            label=int(label)####将string转换为int
            tmpLabel.append(transformLabel(label))】
            fileName = file
            img = cv2.imread(fileName, colorType)
            img_formated = cv2.resize(img, resize_format, interpolation=resize_interpolation)
            img_formated = np.expand_dims(img_formated, axis=0)
            img_flat=dense_result.predict(img_formated)
            img_flat = img_flat.ravel()

            tmpData.append(img_flat)

            if needGenImg:
                genImgs = rotateImg(img_formated)####通过旋转图片进行数据集增广
                for genImg in genImgs:
                    img_flat = img_formated.ravel()
                    tmpData.append(img_flat)
                    tmpLabel.append(transformLabel(label))


    return np.double(tmpData), np.double(tmpLabel)####返回样本和标签

import time
# 获取数据
filePath_train = './crack/train.txt'
filePath_test = './crack/test.txt'
resize_format = (224, 224)####(128, 128)

t1=time.time()
traindata, trainlabel = getDatanew(filePath_train, resize_format, colorType=cv2.IMREAD_COLOR,needGenImg=False)##不需要旋转图片   #####cv2.IMREAD_COLOR   False
t2=time.time()
traindatatime=t2-t1
print("traindatatime is:",traindatatime)
t3=time.time()
testdata, testlabel = getDatanew(filePath_test, resize_format, colorType=cv2.IMREAD_COLOR,needGenImg=False) ###不需要旋转图片     #####cv2.IMREAD_GRAYSCALE
t4=time.time()
testdatatime=t4-t3
print("testdatatime is:",testdatatime)


# 设置参数
N1 = 9 # # of nodes belong to each window 5   {'N1': 2.0, 'N2': 64.0, 'N3': 18278.0
N2 = 32  # # of windows -------Feature mapping layer 7   94   6 48 25514
N3 = 2083 # # of enhancement nodes -----Enhance layer 200 2083
L = 6  # # of incremental steps
M1 = 500  # # of adding enhance nodes
s = 0.8  # shrink coefficient
C = 2 ** -30  # Regularization coefficient
#print(testlabel.shape)###
#print(trainlabel.shape)###
#print(testlabel)

# 跑BLS
print('-------------------BLS_BASE---------------------------')#'N1': 13.0, 'N2': 30.0, 'N3': 26518.0
BLStrain(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)


#################################BayesianOptimization################################
# from bayes_opt import BayesianOptimization
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# import hyperopt.pyll.stochastic
#
# def BLSopt(argsDict):
#
#     N1 = int(argsDict["N1"])
#     N2 = int(argsDict['N2'])
#     N3 = int(argsDict["N3"])
#
#     test_acc=BLStrain(traindata, trainlabel,testdata, testlabel, s, C, N1, N2, N3)
#
#     return -test_acc
#
# spaceBL = {
#     'N1': hp.quniform('N1', 1,10,1),
#     'N2': hp.quniform('N2', 1,60,2),
#     'N3': hp.quniform('N3', 1000,2500,2)
# }
#
# #print(hyperopt.pyll.stochastic.sample(space4knn))
#
# trials = Trials()
# best = fmin(BLSopt, space=spaceBL, algo=tpe.suggest, max_evals=10, trials=trials)
# print('best:',best)
########################################################################################

#Incremental learning
# M1 = 2000 # # of adding new patterns
# print('-------------------BLS_INPUT--------------------------')
# bls_train_input(traindata[0:20000,:],trainlabel[0:20000,:],traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M1)

