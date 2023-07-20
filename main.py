import face_recognition as fr
import cv2
import datetime
import time
import csv

import numpy as np

# capture cam
videoCapture = cv2.VideoCapture(0)
time.sleep(1)

# load images
face1 = fr.load_image_file("faces/face1.jpg")
face2 = fr.load_image_file("faces/face2.jpg")

# encode images
face1Encoding = fr.face_encodings(face1)[0]
face2Encoding = fr.face_encodings(face2)[0]

# make list of known face encodings
knownFaceEncodings = [face1Encoding, face2Encoding]

# make list of known face names
knownFaceNames = ["face1", "face2"]

# copy the known face names into a list of expected students
students = knownFaceNames.copy()

# make 2 empty lists for face locations and face encodings
faceLocations = []
faceEncodings = []

# get the current date and time
currentTime = time.localtime()

# format the time
currentDate = time.strftime("%Y-%m-%d", currentTime)

# make a csv writer
f = open(f"{currentDate}.csv", "w+", newline="")
csvWriter = csv.writer(f)

# make an infinite while loop
while videoCapture.isOpened():
    #   - read the captured cam
    _, frame = videoCapture.read()
    #   - resize the frame
    smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #   - convert to RGB
    rgbSmallFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB)
    #   - get face locations
    faceLocations = fr.face_locations(rgbSmallFrame)
    #   - get face encodings
    faceEncodings = fr.face_encodings(rgbSmallFrame, faceLocations)
    #   - start a for loop
    for fe in faceEncodings:
        #       - get matches: compare face encodings with known face encodings
        matches = fr.compare_faces(knownFaceEncodings, fe)
        #       - get face distance
        faceDistance = fr.face_distance(knownFaceEncodings, fe)
        #       - get best match index
        bestMatchIndex = np.argmin(faceDistance)
        #       - check if best match index exists in matches: if yes then get its name
        name = ""
        if matches[bestMatchIndex]:
            name = knownFaceNames[bestMatchIndex]
        #   - add the text in the frame if the person is present
        if name in knownFaceNames:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness,
                        lineType)
        #   - remove the student from list of expected students and write on the csv file
        if name in students:
            students.remove(name)
            ct = time.strftime("%H:%M:%S", currentTime)
            csvWriter.writerow([name, ct])
    #   - show the frame
    cv2.imshow("Attendance", frame)
    #   - while loop quit logic
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the cam capture
videoCapture.release()

# destroy all windows
cv2.destroyAllWindows()

# close the file
f.close()
