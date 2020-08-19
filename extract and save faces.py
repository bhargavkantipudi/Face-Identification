# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
from numpy import asarray
from keras.models import load_model
import pymongo
import pickle
from bson.binary import Binary

client = pymongo.MongoClient("mongodb+srv://bhargav:5544@cluster0-m9bsr.mongodb.net/facedb?retryWrites=true&w=majority")
mydb = client["facedb"]
mycol = mydb["facedata"]

def extract_face(filename, required_size=(160, 160)):
	try:	
		image = Image.open(filename)
		image = image.convert('RGB')
		pixels = asarray(image)
		detector = MTCNN()
		results = detector.detect_faces(pixels)
		if results==[]:
			return []
		x1, y1, width, height = results[0]['box']
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		face = pixels[y1:y2, x1:x2]
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = asarray(image)
		return face_array
	except:
		print("not able to load ",filename)
		return

def load_faces(directory):
	faces = list()
	for filename in listdir(directory):
		path = directory + filename
		face = extract_face(path)
		if face==[]:
			print("NO face found",path)
			continue
		faces.append(face)
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
