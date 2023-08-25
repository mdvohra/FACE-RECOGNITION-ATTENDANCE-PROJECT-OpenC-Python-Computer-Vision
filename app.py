import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from flask import Flask, render_template, Response, flash, redirect, url_for

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace 'your_secret_key' with a secret key of your choice

path = r'Images'
images = []
recognized_names = []

classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open(r'C:\Users\Moh\PycharmProjects\FaceRecognition\Attendence.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            flash(f'Attendance marked for {name}', 'success')

encodeListKnown = findEncodings(images)

@app.route('/')
def index():
    global recognized_names
    names_to_display = recognized_names
    recognized_names = []  # Clear the recognized_names list for the next cycle
    return render_template('index.html', recognized_names=names_to_display)


def gen():
    global recognized_names
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success or img is None:
            continue

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        if imgS.size == 0:
            continue

        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                recognized_names.append(name)
                y1, x2, y2, x1 = faceLoc
                # ... Rest of the code for drawing the rectangle and displaying the name ...

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_attendance')
def start_attendance():
    flash('Attendance detection started!', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
