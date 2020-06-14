from numpy import expand_dims
from numpy import asarray
from keras.models import load_model
import cv2
import pickle
from pymongo import MongoClient
from bson.binary import Binary
import pymongo

import argparse
parser = argparse.ArgumentParser() 
parser.add_argument("-n", "--Name", help = "Add path for storing images")
args = parser.parse_args() 

name=args.Name

def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]



client = pymongo.MongoClient("mongodb+srv://bhargav:5544@cluster0-m9bsr.mongodb.net/facedb?retryWrites=true&w=majority")
mydb = client["facedb"]
mycol = mydb["facedata"]


model = load_model('model/facenet_keras.h5')
print('Loaded Model')

from PIL import Image
from os import  listdir

def load_faces(directory):
	faces = list()
	for filename in listdir(directory):
		path = directory + filename
		face = Image.open(path)
		faces.append(asarray(face))
	return faces
path="dataset/"+str(name)+"/"
faces=load_faces(path)
data=[]
for face in faces:
    x=get_embedding(model,face)
    mydict = { "name": name, "data":Binary(pickle.dumps(x,protocol=2))}
    data.append(mydict)

x = mycol.insert_many(data)

print(x)
