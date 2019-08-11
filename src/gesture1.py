# -*- coding:utf-8 -*-
#Gesture-recognition-v2
#Author: Kechun Xu
#improved accuracy of the former version

import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt

from PIL import Image
from pylab import *
import math

import time
import multiprocessing as mp

import serial
import serial.tools.list_ports

#检测颜色，修改为手套颜色--HSV
colorLower = np.array([156,43,46])
colorUpper = np.array([180,255,255])

#对于树莓派，1打开电脑摄像头，0打开外部, 对于电脑则相反
camera = 1

def main():

    #img_queues = [mp.Queue(maxsize=2) for _ in camera_ip_l]  # queue

    #img_queues.put(frame) if is_opened else None  # 线程A不仅将图片放入队列
    #img_queues.get() if q.qsize() > 1 else time.sleep(0.01) # 线程A还负责移除队列中的旧图
    
    camera1 = cv.VideoCapture(camera)

    while camera1.isOpened():
        k = cv.waitKey(0)
        ret,frame = camera1.read()
        
        cv.imshow('camera',frame)
        frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
        if k==ord('b') : # 按b捕获图像开始识别,需要修改为指令启动            
            value = gesture_detect(frame)
            print(value)

            '''这里以五种手势表示五个tasks，可以修改'''
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
            else :
                print("Gesture recognition error, please re-identify")

        elif k==ord('q'): # 按q退出
            break
        

    camera1.release()
    cv.destroyAllWindows()

def get_distance(beg, end):#计算两点之间的坐标
    i=str(beg).split(',')
    j=i[0].split('(')
    x1=int(j[1])
    k=i[1].split(')')
    y1=int(k[0])
    i=str(end).split(',')
    j=i[0].split('(')
    x2=int(j[1])
    k=i[1].split(')')
    y2=int(k[0])
    d=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    return d

def gesture_detect(img):#手势识别主函数
    HSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    mask = cv.inRange(HSV,colorLower, colorUpper)

    mask = cv.erode(mask,None,iterations=1)
    mask = cv.dilate(mask,None,iterations=1)
    cv.imshow('mask',mask)
    mask = cv.resize(mask,(0,0),fx=0.5,fy=0.5,interpolation=cv.INTER_NEAREST)

    image,contours,hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    effective_contours = []
    value = []

    if len(contours):
        max_area = cv.contourArea(contours[0])

        for i in range(len(contours)):          
            area_mask = cv.contourArea(contours[i])
            if area_mask > max_area:
                max_area = area_mask
            #print('max_area',max_area)

        for i in range(len(contours)):          
            area_mask = cv.contourArea(contours[i])
            print('area_of_contours',cv.contourArea(contours[i]))
            if area_mask >= 0.3 * max_area:
                effective_contours.append(contours[i])
        #print('effective_contours number', len(effective_contours))

        drawing = np.zeros(img.shape, np.uint8)

        for i in range(len(effective_contours)):

            res = effective_contours[i]
            hull = cv.convexHull(res, returnPoints = False) #得出点集的凸包
            defects = cv.convexityDefects(res, hull) #得出凸缺陷

        #寻找指尖
        ndefects = 0

        if len(defects.shape):
            for j in range(defects.shape[0]):
                s,e,f,_ = defects[j,0]
                beg     = tuple(res[s][0])
                end     = tuple(res[e][0])
                far     = tuple(res[f][0])
                a       = get_distance(beg, end)
                b       = get_distance(beg, far)
                c       = get_distance(end, far)
                angle   = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) 
                    
                if angle <= math.pi/2 :  #计算锐角的个数
                    ndefects = ndefects + 1

                cv.circle(drawing, far, 3, (0,0,255), -1)
                cv.line(drawing, beg, end, (0,0,255), 1)

            value.append(ndefects + 1)
    
            cv.drawContours(img,effective_contours,-1,(0,255,0),1)
            
            img = cv.resize(img,(0,0),fx=0.5,fy=0.5,interpolation=cv.INTER_NEAREST)
            cv.imshow('image',img)
            drawing = cv.resize(drawing,(0,0),fx=0.5,fy=0.5,interpolation=cv.INTER_NEAREST)
            cv.imshow('drawing',drawing)

    return value

main()
