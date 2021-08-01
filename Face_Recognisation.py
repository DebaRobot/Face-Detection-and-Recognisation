# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 23:35:09 2021

@author: h22de
"""

import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Bill Gates', 'Elon Musk']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:/Users/h22de/OneDrive/Desktop/c9u3jvuk_elon-musk-bill-gates_625x300_19_February_20.webp')

gray = cv.cvtColor(img,  cv.COLOR_BGR2GRAY)

#cv.imshow('Person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]
    
    label, confidence =  face_recognizer.predict(faces_roi)
    print(confidence)
    print(people[label])
    
    cv.putText(img, str(people[label]), (50,70), cv.FONT_HERSHEY_COMPLEX, 1.0,
               (0,255,0), 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    
cv.imshow('Detected Face', img)
cv.waitKey(0)
