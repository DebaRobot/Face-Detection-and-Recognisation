# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 22:42:12 2021

@author: Debashis Saha
"""
!pip install opencv-contrib-python --user
import os
import cv2 as cv 
import numpy as np



people = []
for i in os.listdir(r'C:\Users\h22de\OneDrive\Desktop\Great People'):
    people.append(i)
print(people)

DIR = r'C:\Users\h22de\OneDrive\Desktop\Great People'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            
            img_array =  cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1,
                                                       minNeighbors = 4)
            
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
                
create_train()
print("Training Done------------------")
features =  np.array(features, dtype = 'object')
labels =  np.array(labels)

#Train our recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()


##Train the recognizer on features and label list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)


                
            
            
        
