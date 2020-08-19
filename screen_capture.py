import numpy as np
import cv2
import os
import time 


def capture_video(name,img_num):
    path="dataset/"+name
    if not os.path.exists(path):
        os.makedirs(path)
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture("sumo.mp4")
    i=0
    flag=True
    while 1:
        if flag:
            flag=False
            continue
        else:
            flag=True
        ret, img = cap.read()
        #img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_color = img[y:y+h, x:x+w]
            i=i+1
            tmp=path+"/"+str(i)+".jpg"
            resized=cv2.resize(roi_color,(160,160))
            cv2.imwrite(tmp,resized)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,str(i),(15,15),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        cv2.imshow('img',img)
        if i>=img_num:
            break
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

capture_video("sumo",5000)