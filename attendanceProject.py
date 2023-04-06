import cv2
import numpy
import face_recognition
import os
from datetime import datetime
import numpy as np

path = 'pictures'
images = []
classname = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classname.append(os.path.splitext(cl)[0])
print(classname)


def findencoding(images):
    encodelist = []
    for imge in images:
        imge = cv2.cvtColor(imge, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(imge)[0]
        encodelist.append(encode)
    return encodelist


def markAttendance(name):
    with open('newattends.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S:%Y')
            f.writelines(f'\n{name},{dtString}')


encodelistknown = findencoding(images)
print("encoding completed")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodelistknown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classname[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

# faceloc = face_recognition.face_locations(imgone)[0]
# encodeone=face_recognition.face_encodings(imgone)[0]
# cv2.rectangle(imgone,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,255,0),2)

# facelocsam = face_recognition.face_locations(imgsam)[0]
# encodetwo=face_recognition.face_encodings(imgsam)[0]
# cv2.rectangle(imgsam,(facelocsam[3],facelocsam[0]),(facelocsam[1],facelocsam[2]),(255,0,0),2)

# faceDis = face_recognition.face_distance([encodeone],encodetwo)
# comparison = face_recognition.compare_faces([encodeone],encodetwo)

# imgone = face_recognition.load_image_file('pictures/elon.jpg')
# imgone = cv2.cvtColor(imgone,cv2.COLOR_BGR2RGB)

# imgsam = face_recognition.load_image_file('pictures/keer.jpg')
# imgsam = cv2.cvtColor(imgsam,cv2.COLOR_BGR2RGB)
