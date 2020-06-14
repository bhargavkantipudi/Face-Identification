import numpy as np
import cv2
import os
import time 
import argparse
parser = argparse.ArgumentParser() 
parser.add_argument("-p", "--Path", help = "Add path for storing images") 
parser.add_argument("-m", "--No_img", help = "Number of images") 
parser.add_argument("-s", "--Start", help = "Start index") 

args = parser.parse_args() 
path="dataset"
print(args)
#path=input("ENter path")
if args.Path:
    path=args.Path
    print(path)
if not os.path.exists(path):
    os.makedirs(path)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
i=0
if args.Start:
    i=int(args.Start)

while 1:
    ret, img = cap.read()
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
    time.sleep(0.1)
    if int(args.Start):
        if i >= i+int(args.No_img):
            break
    else:
        if i>=int(args.No_img):
            break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        


        
cap.release()
cv2.destroyAllWindows()
 



 #  python .\capture.py -p dataset/bhargav -n 50 -s 0
 