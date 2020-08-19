import numpy as np
import cv2
import os
import time 
from numpy import expand_dims
from numpy import asarray
from keras.models import load_model
import cv2
import pickle
from pymongo import MongoClient
from bson.binary import Binary
import pymongo
import joblib
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pandas as pd
import time 
import joblib
from keras.models import load_model
from numpy import expand_dims
from sklearn.preprocessing import Normalizer

def capture_video(name,img_num):
    path="dataset/"+name
    if not os.path.exists(path):
        os.makedirs(path)
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    i=0

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
        if i>=img_num:
            break
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]



def upload_embedings(name):
    print("started embeding")
    path="dataset/"+name
    if not os.path.exists(path):
        return False
    
    client = pymongo.MongoClient("mongodb+srv://bhargav:5544@cluster0-m9bsr.mongodb.net/facedb?retryWrites=true&w=majority")
    mydb = client["facedb"]
    mycol = mydb["facedata"]
    model = load_model('model/facenet_keras.h5')
    print('Loaded Model')
    data=[]
    for val in os.listdir(path):
        tmp=path+"/"+val
        print("*************",tmp)
        img=cv2.imread(tmp,1)
        x=get_embedding(model,img)
        mydict = { "name": name, "data":Binary(pickle.dumps(x,protocol=2))}
        data.append(mydict)
        print("embeding ",tmp)

    res = mycol.insert_many(data)
    print(res)
def create_model(path="models/"):
    client = pymongo.MongoClient("mongodb+srv://bhargav:5544@cluster0-m9bsr.mongodb.net/facedb?retryWrites=true&w=majority")
    mydb = client["facedb"]
    mycol = mydb["facedata"]
    trainX=[]
    trainy=[]
    for x in mycol.find():
        trainy.append(x["name"])
        trainX.append(pickle.loads(x["data"]))
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    unique=set(trainy)
    print(unique)
    label_dic={}
    labels=out_encoder.inverse_transform(list(unique))
    for x,y in zip(unique,labels):
        label_dic[x]=y
    print(label_dic)
    tmp=path+"face_labels"
    joblib.dump(label_dic, tmp)
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    tmp=path+"face_svm"
    joblib.dump(model, tmp)


def live_detection():
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    facenet_model = load_model('model/facenet_keras.h5')

    loaded_model = joblib.load("model/face_svm")
    label_dict=joblib.load("model/face_labels")
    font=cv2.FONT_HERSHEY_SIMPLEX
    in_encoder = Normalizer(norm='l2')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_color = cv2.resize(img[y:y+h, x:x+w],(160,160))
            fnet_embeding=get_embedding(facenet_model,roi_color)
            #norm_face = in_encoder.transform(np.array(x))
            #print(norm_face)
            sample=[fnet_embeding]
            norm_face = in_encoder.transform(sample)
            yhat_class = loaded_model.predict(norm_face)
            yhat_prob = loaded_model.predict_proba(norm_face)
            # get name
            class_index = yhat_class[0]
            class_probability = yhat_prob[0,class_index] * 100
            print("class"+label_dict[class_index]+"-")
            color=(255,0,0)
            if label_dict[class_index]=="unknown":
                color=(0,0,255)
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            txt=label_dict[class_index]+str(round(class_probability, 2))
            cv2.putText(img,txt,(x,y),font,0.8,(0,255,0),1)   
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()


