import numpy as np

import cv2
import controller_
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA # Principal component analysis

names={0:'Mask',1:'No Mask'}

with_mask=np.load('with_mask.npy')
without_mask=np.load('without_mask.npy')

with_mask=with_mask.reshape(200,50*50*3)  # 4 dimensional into a 2 Dimensional
without_mask=without_mask.reshape(200,50*50*3)

X=np.r_[with_mask,without_mask]

labels=np.zeros(X.shape[0])  # will give 200
labels[200:]=1.0

# SuperVised  and Unsupervised Learning ---- a machine that learn from the experience 
#Semi- Supervised and Reinforcement Learning

# Supervised 
"""
Regression
 
"""
x_train,x_test,y_train,y_test=train_test_split(X,labels,test_size=0.25)# 25 % reserved for testing other ML

#Dimensionality Reduction
pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
svm=SVC()
svm.fit(x_train,y_train)

y_pred=svm.predict(x_test)
font=cv2.FONT_HERSHEY_SIMPLEX
#***************************************************************
capture=cv2.VideoCapture(0)
haar_data=cv2.CascadeClassifier('Resources\haarcascade_frontalface_default.xml')
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
            face=img[y:y+h,x:x+w,:]  # to slice the face
            face=cv2.resize(face,(50,50))
            face=face.reshape(1,-1)
            face=pca.transform(face)
            pred=svm.predict(face)[0]
            print("if  mask then it is ",int(pred))
            n=names[int(pred)]
            cv2.putText(img,n,(x,y),font,1,(244,250,250),2)
        cv2.imshow('Result',img)
        if cv2.waitKey(1) == 27:  #27 ASCII number of Esc
            break
#***************************************************************

capture.release()
cv2.destroyAllWindows()