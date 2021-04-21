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

##阈值分割
def otsu_binarization(img, th=128):
    H, W = img.shape
    out = img.copy()

    max_sigma = 0
    max_t = 0

    # determine threshold
    for _t in range(1, 255):
        v0 = out[np.where(out < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)
        v1 = out[np.where(out >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    # Binarization
    print("threshold >>", max_t)
    th = max_t
    out[out < th] = 0
    out[out >= th] = 255

    return out

import numpy as np
import matplotlib.pyplot as pylab
from skimage.filters import sobel, threshold_otsu
from skimage.feature import canny
import cv2
import matplotlib.pyplot as plt
###二值化处理
def seg(img):
    # 变微灰度图
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 中值滤波
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
    ##ResNet50作为卷积模块
    model = ResNet50(include_top=True,weights='G:/zy/01/resnet50_weights_tf_dim_ordering_tf_kernels.h5')  ##imagenet #####自动的下载权重   include_top=True代表包含最后的全连接层
    dense_result = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)  ####分类层的前一层

    for i in range(1, 2):  ###(1,3)是指【1,2】
        image = cv2.imread('./demo/width/%d.jpg'%(i),cv2.IMREAD_COLOR)

        #
        w = image.shape[1]
        h = image.shape[0]
        # 本代码将图片分为3×3，共九个子区域，winW, winH和stepSize可自行更改,按照比例计算的，所以不涉及
        (winW, winH) = (int(w/8),int(h/8))######8
        stepSize = (int(w/8), int(h/8))#####步长这么设置，各个滑动窗口之间是没有重合的
        #print(winW, winH)
        cnt = 0
        clone = image.copy()  ##这个是整个图片
        # create a mask创造一个与原图片一样大小的蒙版，并将黑色0的变成白色255的
        mask = np.zeros(image.shape[:2], np.uint8)+255####如果直接存储图片image.shape[:3],如果经过二值化处理image.shape[:2]
        #mask[:, 0] = np.zeros(w) + 255
        #mask[:, 1] = np.ones(h) + 255

        for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # since we do not have a classifier, we'll just draw the window 展示滑动的过程

            #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)##在图片上绘制一个矩形 左上角坐标，右下角坐标，颜色，线宽
            #cv2.imshow("Window", clone)
            #cv2.waitKey(1000)

            slice = image[y:y+winH,x:x+winW]###滑动窗口中的图片
            #print(slice)
            resize_format=(224, 224)###
            #cv2.imwrite("G:/zy/0/crack/total/" + str(i)+"-"+str(cnt) + "slice.jpg", slice)##保存,不能有中文路径
            resize_interpolation = cv2.INTER_NEAREST
            img_formated = cv2.resize(slice, resize_format,interpolation=resize_interpolation)
            ##经过预处理
            img_formated = np.expand_dims(img_formated, axis=0)
            img_flat = dense_result.predict(img_formated)
            img_flat = img_flat.ravel()

            tmpData = []
            tmpData.append(img_flat)
            tmpData=np.double(tmpData)
            ##输出BLS进行分类
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
                #cv2.imwrite("G:/zy/01/demo/tuya/" + str(cnt) + "Tlice.jpg", slice)  ##保存,不能有中文路径 这是检测图放在蒙版上
                mask[y:y+winH,x:x+winW]=slice ##image[y:y+winH,x:x+winW]###将图片中检测出来的部分放在与图片相同大小蒙版的对应位置

            #cv2.namedWindow('sliding_slice',0)
            #cv2.imshow('sliding_slice', slice)
            #cv2.waitKey(1000)
            cnt = cnt + 1
        #cv2.imshow("Window", clone)
        #cv2.imshow("Window2", mask)
        #cv2.waitKey(0)
        cv2.imwrite("G:/zy/01/demo/width/" + str(i)+"-"+str(cnt) + "clone.jpg", clone)
        cv2.imwrite("G:/zy/01/demo/width/" + str(i) + "-" + str(cnt) + "seg.jpg", mask)