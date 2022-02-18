# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 22:32:25 2021

@author: User
"""
#使用OpenCV 讀取YOLOv3模型
#! git clone https://github.com/pjreddie/darknet.git
import cv2
import numpy as np

import wget
site_url="https://pjreddie.com/media/files/yolov3.weights"
file_name = wget.download(site_url)
print(file_name)

weightsPath="yolov3.weights"
configPath="yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)

layer_names = net.getLayerNames()
#print(layer_names)

output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(output_layers)

#讀取相關參數
classes = [line.strip() for line in open("C:/Users/User/OneDrive/桌面/機器學習/期末報告/obj.names")]
colors = [(0,0,255),(255,0,0),(0,255,0)]

#讀取圖片
from PIL import Image
Image.open("C:/Users/User/OneDrive/桌面/機器學習/medical-masks-dataset/images/71a899b4-912e-426e-9694-7d84f0bc42ca.jpg")
img = cv2.imread("C:/Users/User/OneDrive/桌面/機器學習/medical-masks-dataset/images/71a899b4-912e-426e-9694-7d84f0bc42ca.jpg")
#print(img.shape)

#利用YOLOv3 模型辨識圖片
img2 = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img2.shape 
blob = cv2.dnn.blobFromImage(img2, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

print(len(outs))
print(outs[0].shape)

#擷取偵測物件位置
class_ids = []
confidences = []
boxes = []
    
for out in outs:
    for detection in out:
        tx, ty, tw, th, confidence = detection[0:5]
        scores = detection[5:]
        class_id = np.argmax(scores)  
        if confidence > 0.3:   
            center_x = int(tx * width)
            center_y = int(ty * height)
            w = int(tw * width)
            h = int(th * height)
            
            # 取得箱子方框座標
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
print(len(boxes))
#non-maxima suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

#框住偵測物件區域
font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img2, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img2, label, (x, y - 5), font, 2, color, 3)
#print(img2.shape)
#%pylab inline
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

#將辨識過程包裝成函數
import cv2
import numpy as np
weightsPath2="C:/Users/User/OneDrive/桌面/機器學習/期末報告/yolov3_1100.weights"
configPath2="C:/Users/User/OneDrive/桌面/機器學習/期末報告/yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath2,weightsPath2)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = [line.strip() for line in open("C:/Users/User/OneDrive/桌面/機器學習/期末報告/obj.names")]
colors = [(0,0,255),(255,0,0),(0,255,0)]

def yolo_detect(frame):
    # forward propogation
    img3 = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = img3.shape 
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # get detection boxes
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            tx, ty, tw, th, confidence = detection[0:5]
            scores = detection[5:]
            class_id = np.argmax(scores)  
            if confidence > 0.3:   
                center_x = int(tx * width)
                center_y = int(ty * height)
                w = int(tw * width)
                h = int(th * height)

                # 取得箱子方框座標
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # draw boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img3, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img3, label, (x, y -5), font, 3, color, 3)
    return img3

#測試函數功能
img4 = cv2.imread("C:/Users/User/OneDrive/桌面/機器學習/期末報告/b.jpg")
im = yolo_detect(img4)
img_rgb = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

#使用攝像頭即時偵測物件
import cv2
import imutils
import time

VIDEO_IN = cv2.VideoCapture(0)


while True:
    hasFrame, frame = VIDEO_IN.read()
    
    img5 = yolo_detect(frame)
    cv2.imshow("Frame", imutils.resize(img5, width=850))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
VIDEO_IN.release()
cv2.destroyAllWindows()