import cv2 as cv
img = cv.imread('C:/Users/h22de/OneDrive/Desktop/istockphoto-1146473249-612x612.jpg')
cv.imshow('Person', img)

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grey Person', grey)


##Reading the file
haar_cascade = cv.CascadeClassifier('haar_face.xml')

#Read the instance of class- Detect Faces
#List of all coordinate of the faces detect in an image
faces_rect = haar_cascade.detectMultiScale(grey, scaleFactor =  1.1, minNeighbors = 5)

print("The number of the faces founded ",len(faces_rect))

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)
    
cv.imshow("Detected Faces", img)


cv.waitKey(0)
