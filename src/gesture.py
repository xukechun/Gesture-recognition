# -*- coding:utf-8 -*-
import cv2 as cv
import cv2
import numpy as np 
from matplotlib import pyplot as plt

from PIL import Image
from pylab import *
import math

import time

import serial
import serial.tools.list_ports

#检测颜色，修改为手套颜色--hsv
colorLower = np.array([156,43,46])
colorUpper = np.array([180,255,255])

#1打开电脑摄像头，0打开外部
camera = 1

def main():
    camera1 = cv.VideoCapture(camera)

    while camera1.isOpened():
        k = cv.waitKey(10)
        ret,frame = camera1.read()
        
        cv.imshow('camera',frame)
        #frame = cv.resize(frame, (0, 0), fx=0.9, fy=0.9, interpolation=cv.INTER_NEAREST)
        if k==ord('b') : # 按b捕获图像开始识别            
            value = gesture_detect(frame)
            print(value)

            '''这里以五种手势表示五个tasks，可以修改'''
            """手势识别1和2比较稳定，0不太稳定"""
            if value == [1]:
                print("task1")
            elif value == [2]:
                print("task2")
            elif value == [1,1]:
                print("task3")
            elif value == [1,2] or value == [2,1]:
                print("task4")  
            elif value == [2,2]:
                print("task5")
            #elif value == [2]:
            #    print("task6")
            else :
                print("Gesture recognition error, please re-identify")

        elif k==ord('q'): # 按q退出
            break
        

    camera1.release()
    cv.destroyAllWindows()


def gesture_detect(img):
    HSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    mask = cv.inRange(HSV,colorLower, colorUpper)

    mask = cv.erode(mask,None,iterations = 1)
    mask = cv.dilate(mask,None,iterations = 1)
    cv.imshow('mask',mask)
    mask = cv.resize(mask, (0, 0), fx=0.9, fy=0.9, interpolation=cv.INTER_NEAREST)

    image,contours,hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    effective_contours = []
    value = []

    max_area = cv.contourArea(contours[0])
    for i in range(len(contours)):          
        area_mask = cv.contourArea(contours[i])
        if area_mask > max_area:
            max_area = area_mask
        print('max_area',max_area)
    for i in range(len(contours)):          
        area_mask = cv.contourArea(contours[i])
        print('area_of_contours',cv.contourArea(contours[i]))
        if area_mask >= 0.3 * max_area:
            effective_contours.append(contours[i])
    print('effective_contours number', len(effective_contours))
    drawing = np.zeros(img.shape, np.uint8)

    for i in range(len(effective_contours)):
        res = effective_contours[i]
        hull = cv.convexHull(res) #得出点集的凸包

        cv.drawContours(drawing, [res], 0, (0, 255, 0), 2)   #画出区域轮廓
        cv.drawContours(drawing, [hull], 0, (0, 0, 255), 3)  #画出凸包轮廓
        
        moments = cv.moments(res) #求区域轮廓各阶矩
        center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
        cv.circle(drawing, center, 8, (0,0,255), -1)   #画出重心

        fingerRes = [] #寻找指尖
        max = 0;min = 999999999;count = 0; notice = 0; cnt = 0; average = 0
        for j in range(len(res)):
            temp = res[j]
            dist = math.sqrt((temp[0][0]-center[0])*(temp[0][0] - center[0]) + (temp[0][1] - center[1])*(temp[0][1] - center[1]))
            average += dist/len(res)
        #print('average',average)

        for j in range(len(res)):
            temp = res[j]
            dist = math.sqrt((temp[0][0]-center[0])*(temp[0][0] - center[0]) + (temp[0][1] - center[1])*(temp[0][1] - center[1]))
            #print('dist = ',dist)
            if dist < min:
                min = dist
            if dist > max:
                max = dist
                notice = j
            if dist != max and (dist - min) > min*0.5:
                count += 1
                if count > 40:
                    flag = False
                    if center[1] < res[notice][0][1]:#去除手心下的凸包
                        continue
                    for k in range(len(fingerRes)): #离得太近不算
                        if abs(temp[0][0]-fingerRes[k][0]) < 50:
                            flag = True
                            break
                    if flag :
                        continue
                    
                    if max > 0.8*average:
                        fingerRes.append(res[notice][0])
                        cv.circle(drawing, tuple(res[notice][0]), 8 , (255, 0, 0), -1) #画出指尖
                        cv.line(drawing, center, tuple(res[notice][0]), (255, 0, 0), 2)
                        cnt += 1
                    
                    count = 0
                    max = 0
                    min = 99999999

        value.append(cnt)
    

    cv.drawContours(img,effective_contours,-1,(0,255,0),1)
    
    cv.imshow('imag',img)
    img = cv.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
    cv.imshow('drawing',drawing)
    drawing = cv.resize(drawing, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
    
    return value

main()
