# **Gusture-recognition**

手势识别

本项目识别的是红手套，可以根据识别对象修改HSV范围

```python
colorLower = np.array([156,43,46])
colorUpper = np.array([180,255,255])
```

摄像头说明

```python
#1打开电脑摄像头，0打开外部，若是树莓派则1为外部摄像头
camera = 1
```

## gesture

采用距离手掌心的距离进行检测，主函数为`gesture_detect`，主要通过检测凸包，根据离手掌心的距离判断是否为指尖

```python
def gesture_detect(img):
    HSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    mask = cv.inRange(HSV,colorLower, colorUpper)

    mask = cv.erode(mask,None,iterations = 1)  #迭代次数可以根据识别距离调整，识别距离大时，为保证清晰度，应减小迭代次数
    mask = cv.dilate(mask,None,iterations = 1)
    cv.imshow('mask',mask)
    mask = cv.resize(mask, (0, 0), fx=0.9, fy=0.9, interpolation=cv.INTER_NEAREST)

    image,contours,hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    effective_contours = []
    value = []

    max_area = cv.contourArea(contours[0])
    for i in range(len(contours)):          
        area_mask = cv.contourArea(contours[i])
        if area_mask > max_area: #寻找最大轮廓面积，根据最大面积丢弃一定的轮廓防止误识别
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
        hull = cv.convexHull(res) 	#得出点集的凸包

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
```



## gesture1

主要通过检测凹陷，检测锐角的个数实现手势识别，对gesture进行了改进

最主要的修改如下：

```python
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
```

## gesture2

gesture1比gesture算法鲁棒性高，但是电脑可能会出现摄像头延迟2-3帧的问题，目前此版本仍存在bug，待更新

主要思想是通过多线程避免延时