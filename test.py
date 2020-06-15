from allFunctions import create_model
create_model("model/")




face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
facenet_model = load_model('model/facenet_keras.h5')

loaded_model = joblib.load("model/face_svm")
label_dict=joblib.load("model/face_labels")
font=cv2.FONT_HERSHEY_SIMPLEX
in_encoder = Normalizer(norm='l2')
cap = cv2.VideoCapture(0)

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
        print("class",label_dict[class_index])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        txt=label_dict[class_index]+str(round(class_probability, 2))
        cv2.putText(img,txt,(x,y),font,1,(0,255,0),1)   
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()