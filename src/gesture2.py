# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt

from PIL import Image
from pylab import *
import math

import time

import serial
import serial.tools.list_ports

'''此版本旨在采用多线程解决摄像头的延时问题，由于不同的电脑延时的情况不一，也可能没有延时，另外本版本仍然存在bug，后续会更新'''

#检测颜色，修改为手套颜色--HSV
colorLower = np.array([156,43,46])
colorUpper = np.array([180,255,255])

#对于树莓派，1打开电脑摄像头，0打开外部, 对于电脑则相反
camera = 1

class Stack:
 
    def __init__(self, stack_size):
        self.items = []
        self.stack_size = stack_size
 
    def is_empty(self):
        return len(self.items) == 0
 
    def pop(self):
        return self.items.pop()
 
    def peek(self):
        if not self.isEmpty():
            return self.items[len(self.items) - 1]
 
    def size(self):
        return len(self.items)
 
    def push(self, item):
        if self.size() >= self.stack_size:
            for i in range(self.size() - self.stack_size + 1):
                self.items.remove(self.items[0])
        self.items.append(item)

def capture_thread(camera1, frame_buffer, lock):
    print("capture_thread start")
    vid = cv2.VideoCapture(camera1)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    while True:
        return_value, frame = vid.read()
        if return_value is not True:
            break
        lock.acquire()
        frame_buffer.push(frame)
        lock.release()
        cv2.waitKey(25)

def play_thread(frame_buffer, lock):
    print("detect_thread start")
    print("detect_thread frame_buffer size is", frame_buffer.size())

    while True:
        if frame_buffer.size() > 0:
            lock.acquire()
            frame = frame_buffer.pop()
            lock.release()
            if k==ord('b') : # 按b捕获图像开始识别,需要修改为指令启动            
            value = gesture_detect(frame)
            print(value)

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

def main():
    Stack cap
    capture_thread(camera=1,cap,lock)

    while camera1.isOpened():
        k = cv.waitKey(0)
        ret,frame = camera1.read()
        
        cv.imshow('camera',frame)
        frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
        
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

        for j in range(defects.shape[0]):
            s,e,f,_ = defects[j,0]
            beg     = tuple(res[s][0])
            end     = tuple(res[e][0])
            far     = tuple(res[f][0])
            a       = get_distance(beg, end)
            b       = get_distance(beg, far)
            c       = get_distance(end, far)
            angle   = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) # * 57
                
            if angle <= math.pi/2 :
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
