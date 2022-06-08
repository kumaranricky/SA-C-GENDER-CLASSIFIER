# <p align="center"> Skill Assessment</p>
# <p align="center"> GENDER CLASSIFIER</p>
# Algorithm
1.Import necessary packages.

2.Read the image and identify  the face in image using opencv.

3.Form the CNN model using the necessary layers and filters.

4.Train the model and save in the caffemodel format.

5.convert the caffemodel to  create the prototext(prototype machine learning model).

6.Test the model upon various humans images.

7.Display th output Whether the image male or female

## Program:
```
/*
Program to implement 
Developed by   : Kumaran B
RegisterNumber :  212220230026
*/
```
```python

import cv2
import math
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes
    
   faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"

genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)

genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)


genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture("gh.jpg")
padding=20
while cv2.waitKey(1)<0 :
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')
        cv2.putText(resultImg, f'{gender}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        plt.axis("off")
        plt.title('Original image')
        plt.imshow(resultImg)
```

## OUTPUT:
![Screenshot (287)](https://user-images.githubusercontent.com/75243072/172646128-50430bbc-480a-43d5-b41f-39d86f304cd0.png)
![Screenshot (288)](https://user-images.githubusercontent.com/75243072/172646198-32009acc-d990-4dd0-b4ca-8b92d92c416c.png)
