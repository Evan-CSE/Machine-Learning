# Importing all required packages
import cv2
import numpy as np
import matplotlib.pyplot as plt 
# %matplotlib inline

body_cascade = cv2.CascadeClassifier('frontalFace.xml')
cap = cv2.VideoCapture('completed.mp4')

net = cv2.dnn.readNet('dnn_model/yolov4.weights', 'dnn_model/yolov4.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(608, 608), scale=1/255)

classes = []
with open('dnn_model/classes.txt', "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

count=0
frameCount = 0
frameTakenCount = 0
while(cap.isOpened()):
    frameCount+=1
    ret, frame = cap.read()
    

    if(frameCount%12):
        continue

    frame = cv2.resize(frame,(608,608))
    if ret == True:
        # body_rect = body_cascade.detectMultiScale(frame, scaleFactor = 1.15, minNeighbors = 5)   
        #print(len(body_rect))

        # for (x, y, w, h) in body_rect:
        #     cv2.rectangle(frame, (x, y),
        #                 (x + w, y + h), (255, 255, 255), 10)  
        #     cropped_img = frame[y:y+h,x:x+w]
        #     #cv2.imwrite("frames/%#05d.jpg" % (count+1), cropped_img)
        #     # cv2.imshow('Frame', cropped_img)
        #     print(cropped_img.shape)
        #     count+=1
        # print(frame.shape)
        class_ids, scores, boxes = model.detect(frame, nmsThreshold=0.4, confThreshold=0.5)
        for idx in range(len(boxes)):
            if(class_ids[idx] == 0):
                x, y, w, h = boxes[idx]
                cropped = frame[y:y+h, x:x+w]
                # cropped_img = cv2.resize(cropped, (227, 227))
                cv2.imwrite("frames/%#05d.jpg" % (count+1), cropped)
                count += 1
                # print("saved img no = ", count)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        frameTakenCount += 1
        print(frameTakenCount)
        if(frameTakenCount==36000):
            break
        # frame = cv2.resize(frame,(1580,900))
        # cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
