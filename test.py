import cv2
import numpy as np
import pytesseract
from cnocr import CnOcr
import cv2
from cnstd import CnStd
#rawImage = cv2.imread("3.jpg")
kernel = np.ones((10,10),np.uint8)

global std
global cn_ocr
std = CnStd(model_name='resnet50_v1b', root='C:/Users/dell/AppData/Roaming/cnstd/0.1.0/mobilenetv3')
cn_ocr = CnOcr(model_name="conv-lite-fc")

#颜色提取方法
def color_identify(name):
    rawImage = cv2.imread(name)
    original = cv2.imread(name)
    img = cv2.imread(name)
    # 蓝色的范围
    lower_blue = np.array([70, 110, 110])
    upper_blue = np.array([130, 255, 255])

    #中值滤波
    hsv = cv2.medianBlur(rawImage,5)

    #转为hsv
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    #颜色提取
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = mask_blue
    res = cv2.bitwise_and(rawImage, rawImage, mask=mask)
    res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

    #二值化
    ret, image = cv2.threshold(res, 0, 255,cv2.THRESH_OTSU)

    #查找轮廓
    contours,hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for item in contours:
        x,y,weight,height = cv2.boundingRect(item)
        if weight > (height * 2) and weight > 100 and height > 100:
            # 裁剪区域图片
            cv2.rectangle(image, (x, y), (x + weight, y + height), (0, 255, 0), 2)
            chepai = original[y:y + height, x:x + weight]
            list = identify(chepai)
            print(list)
            if(len(list) > 3):
                cv2.resize(chepai,(300,100))
                #cv2.imshow('chepai'+str(x), chepai)
                cv2.imwrite("pai.jpg",chepai)
                return list

    return [0]

#图像处理方法
def picture_identify(name):
    original=cv2.imread(name)
    rawImage = cv2.imread(name)
    #高斯模糊
    image = cv2.GaussianBlur(rawImage, (3, 3), 0)
    # image = cv2.medianBlur(rawImage,5)
    #image = cv2.pyrMeanShiftFiltering(rawImage,21,51)
    #转为灰度图像
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #拉普拉斯算子处理
    #image = cv2.Laplacian(image,cv2.CV_64F)

    #Canny边缘检测
    image = cv2.Canny(image,100,300)

    #Sobel算子
    Sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    # Sobel_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(Sobel_x)  # 转回uint8
    # absY = cv2.convertScaleAbs(Sobel_y)
    # dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    image = absX

    #图像二值化
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    #闭操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX,iterations = 1)
    #
    # #膨胀腐蚀

    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))

    image = cv2.dilate(image, kernelX)
    image = cv2.erode(image, kernelX)

    image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)

    #中值滤波
    image = cv2.medianBlur(image, 15)


    #查找轮廓
    contours,hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for item in contours:
        x,y,weight,height = cv2.boundingRect(item)
        if weight > (height * 2) and weight > 50 and height > 25:
            # 裁剪区域图片
            cv2.rectangle(image, (x, y), (x + weight, y + height), (0, 255, 0), 2)
            chepai = original[y:y + height, x:x + weight]
            list = identify(chepai)
            print(list)
            if (len(list) > 3):
                cv2.resize(chepai, (300, 100))
                # cv2.imshow('chepai'+str(x), chepai)
                cv2.imwrite("pai.jpg", chepai)
                return list
    return [0]


#检测车牌号
def identify(image):
    std = CnStd(model_name='resnet50_v1b', root='C:/Users/dell/AppData/Roaming/cnstd/0.1.0/mobilenetv3')
    global cn_ocr
    ocr_res = []
    box_info_list = std.detect(image)

    ocr_res=[]
    for box_info in box_info_list:
        cropped_img = box_info['cropped_img']
        ocr_res.extend(cn_ocr.ocr_for_single_line(cropped_img))
        print('ocr result1: %s' % ''.join(ocr_res))
    return ocr_res