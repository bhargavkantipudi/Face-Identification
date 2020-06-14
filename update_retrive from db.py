

import pymongo
import pickle
from bson.binary import Binary

client = pymongo.MongoClient("mongodb+srv://bhargav:5544@cluster0-m9bsr.mongodb.net/facedb?retryWrites=true&w=majority")
mydb = client["facedb"]
mycol = mydb["facedata"]
#mydict = { "name": "bhargav", "data":Binary(pickle.dumps("test"))}

#x = mycol.insert_one(mydict)

print(x.inserted_id)
for x in mycol.find():
    y=pickle.loads(x["data"])
    print(y)