import cv2
import numpy as np
import os

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[1]):
        for x in range(0, image.shape[1], stepSize[0]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



from skimage import data,filters
##腐蚀
def Morphology_Erode(img, Erode_time=1):
    H, W = img.shape
    out = img.copy()
    # kernel
    MF = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.int)
    # each erode
    for i in range(Erode_time):
        tmp = np.pad(out, (1, 1), 'edge')
        # erode
        for y in range(1, H):
            for x in range(1, W):
                if np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) < 255 * 4:
                    out[y, x] = 0

    return out


import numpy as np
import matplotlib.pyplot as pylab
from skimage.filters import sobel, threshold_otsu
from skimage.feature import canny
import cv2
import matplotlib.pyplot as plt
###
def seg(img):
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    GrayImage = cv2.medianBlur(GrayImage, 5)
    ret, th2 = cv2.threshold(GrayImage, 80, 255, cv2.THRESH_BINARY)
    #cv2.imshow('binary', th2)
    #cv2.waitKey(0)
    return th2


from BroadLearningSystem import BLStrain,BLStest,BLStestone
from keras.models import Sequential,Model,Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
#from keras.applications import vgg16, AlexNet#inception_v3, resnet50, mobilenet

if __name__ == '__main__':

    model = ResNet50(include_top=True,weights='G:/zy/01/resnet50_weights_tf_dim_ordering_tf_kernels.h5')  ##imagenet #####自动的下载权重   include_top=True代表包含最后的全连接层
    dense_result = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)  ####分类层的前一层

    for i in range(1, 2):
        image = cv2.imread('./demo/width/%d.jpg'%(i),cv2.IMREAD_COLOR)

        #
        w = image.shape[1]
        h = image.shape[0]
        #
        (winW, winH) = (int(w/8),int(h/8))
        stepSize = (int(w/8), int(h/8))
        #print(winW, winH)
        cnt = 0
        clone = image.copy()  #
        #
        mask = np.zeros(image.shape[:2], np.uint8)+255##

        for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
            #
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            #
            #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)#
            #cv2.imshow("Window", clone)
            #cv2.waitKey(1000)

            slice = image[y:y+winH,x:x+winW]###
            #print(slice)
            resize_format=(224, 224)###
            #cv2.imwrite("G:/zy/0/crack/total/" + str(i)+"-"+str(cnt) + "slice.jpg", slice)##
            resize_interpolation = cv2.INTER_NEAREST
            img_formated = cv2.resize(slice, resize_format,interpolation=resize_interpolation)
            ##
            img_formated = np.expand_dims(img_formated, axis=0)
            img_flat = dense_result.predict(img_formated)
            img_flat = img_flat.ravel()

            tmpData = []
            tmpData.append(img_flat)
            tmpData=np.double(tmpData)

            N1=9    ###这些参数要与训练时的参数一致
            N2=32
            N3=2083
            s=0.8
            C=2 ** -30
            time,label=BLStestone(tmpData, s, C, N1, N2, N3)
            #print(label)
            if label == 1:
                cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 0), 3)
                slice=seg(slice)
                #cv2.imwrite("G:/zy/01/demo/tuya/" + str(cnt) + "Tlice.jpg", slice)
                mask[y:y+winH,x:x+winW]=slice #

            #cv2.namedWindow('sliding_slice',0)
            #cv2.imshow('sliding_slice', slice)
            #cv2.waitKey(1000)
            cnt = cnt + 1
        #cv2.imshow("Window", clone)
        #cv2.imshow("Window2", mask)
        #cv2.waitKey(0)
        cv2.imwrite("G:/zy/01/demo/width/" + str(i)+"-"+str(cnt) + "clone.jpg", clone)
        cv2.imwrite("G:/zy/01/demo/width/" + str(i) + "-" + str(cnt) + "seg.jpg", mask)