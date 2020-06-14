from numpy import expand_dims
from numpy import asarray
from keras.models import load_model
import cv2
import pickle
from pymongo import MongoClient
from bson.binary import Binary
import pymongo

client = pymongo.MongoClient("mongodb+srv://bhargav:5544@cluster0-m9bsr.mongodb.net/facedb?retryWrites=true&w=majority")
mydb = client["facedb"]
mycol = mydb["facedata"]


def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]
model = load_model('model/facenet_keras.h5')
print('Loaded Model')
img=cv2.imread("dataset/bhargav/1.jpg",1)
x=get_embedding(model,img)
print(type(x))
mydict = { "name": "bhargav", "data":Binary(pickle.dumps(x,protocol=2))}

x = mycol.insert_one(mydict)

print(x.inserted_id)
print("inserting with cpickle protocol 2")
print("reading cpickle protocol 2")
#%timeit -n 100 [cPickle.loads(x['cpickle']) for x in collection.find()]
for x in mycol.find():
    print(x)