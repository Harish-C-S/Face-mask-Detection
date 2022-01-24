import cv2
import numpy as np
import matplotlib.pyplot as plt
path='Resources\lena.png'
img=cv2.imread('C:\Harish\Programming_Tranformation\Pythoner\OPENCV_py\Face Mask Detection\Resources\lena.png')
#print(img.shape)
#img[0]=1
capture=cv2.VideoCapture(0)
data=[]
haar_data=cv2.CascadeClassifier('Resources\haarcascade_frontalface_default.xml')
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
            face=img[y:y+h,x:x+w,:]  # to slice the face
            face=cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<400:
                data.append(face)
        cv2.imshow('Result',img)
        if cv2.waitKey(1) == 27 or len(data)>=200: #27 ASCII number of Esc
            break

np.save('with_mask.npy',data)









capture.release()        
cv2.destroyAllWindows()
