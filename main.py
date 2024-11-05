import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'uploads'
images = []
classNames = []
myList = os.listdir(path)
print("Training Images:", myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("Class Names:", classNames)



class attendance():
    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                encodeList.append(encodings[0])  
        return encodeList


    def markAttendance(name):
        with open('Attendence.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = [line.split(',')[0] for line in myDataList]

            if name not in nameList:  
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
                print(f"Attendance marked for {name}")
                print("Success")
            else:
                print(f"{name} is already marked present")


    encodeListKnown = findEncodings(images)
    print("Encoding Complete")


    current_path = 'attendance_uploads'
    current_images = os.listdir(current_path)
    print("Current Images:", current_images)


    for current_img_name in current_images:
        current_img_path = f'{current_path}/{current_img_name}'
        current_img = cv2.imread(current_img_path)
        current_img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
    
    
        facesCurFrame = face_recognition.face_locations(current_img_rgb)
        encodesCurFrame = face_recognition.face_encodings(current_img_rgb, facesCurFrame)

        for encodeFace in encodesCurFrame:
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        
            matchIndex = np.argmin(faceDis)

       
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                markAttendance(name)
            else:
                print(f"No match found for {current_img_name}")
                print("Failed")

    print("Attendance marking process complete.")


