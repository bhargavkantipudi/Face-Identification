# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
import cv2
from numpy import asarray
from keras.models import load_model
import pymongo
import pickle
from bson.binary import Binary

client = pymongo.MongoClient("mongodb+srv://bhargav:5544@cluster0-m9bsr.mongodb.net/facedb?retryWrites=true&w=majority")
mydb = client["facedb"]
mycol = mydb["facedata"]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def load_faces(directory):
	faces = list()
	for filename in listdir(directory):
		path = directory + filename
        image = cv2.imread(filename,1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        x,y,w,h= faces[0]
        roi_color = img[y:y+h, x:x+w]
        resized=cv2.resize(roi_color,required_size)
		faces.append(resized)
	return faces

def load_dataset(directory):
	X, y = list(), list()
	for subdir in listdir(directory):
		path = directory + subdir + '/'
		if not isdir(path):
			continue
		faces = load_faces(path)
		labels = [subdir for _ in range(len(faces))]
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)
def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

# load train dataset
trainX, trainy = load_dataset('raw_dataset/')
model = load_model('model/facenet_keras.h5')
print('Loaded Model')
newTrainX = list()
data=[]
for face_pixels,label in zip(trainX,trainy):
	embedding = get_embedding(model, face_pixels)
	mydict = { "name": label, "data":Binary(pickle.dumps(embedding,protocol=2))}
	data.append(mydict)
res = mycol.insert_many(data)

print(res)











