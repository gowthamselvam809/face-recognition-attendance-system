import cv2
import numpy
import face_recognition

imgone = face_recognition.load_image_file('pictures/bill1.jpg')
imgone = cv2.cvtColor(imgone,cv2.COLOR_BGR2RGB)
imgsam = face_recognition.load_image_file('pictures/elon0.jpg')
imgsam = cv2.cvtColor(imgsam,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgone)[0]
encodeone=face_recognition.face_encodings(imgone)[0]
cv2.rectangle(imgone,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,255,0),2)

facelocsam = face_recognition.face_locations(imgsam)[0]
encodetwo=face_recognition.face_encodings(imgsam)[0]
cv2.rectangle(imgsam,(facelocsam[3],facelocsam[0]),(facelocsam[1],facelocsam[2]),(255,0,0),2)

faceDis = face_recognition.face_distance([encodeone],encodetwo)

comparison = face_recognition.compare_faces([encodeone],encodetwo)
print(comparison, faceDis)
cv2.putText(imgone,f'{comparison} {round(faceDis[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)



#cv2.imshow('gowtham',imgone)
#cv2.imshow('gow samp',imgsam)
#cv2.waitKey(0)