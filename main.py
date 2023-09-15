import dlib
import cv2
import time
import math
from ubidots import ApiClient
import speech_recognition as sr
import RPi.GPIO as GPIO
from gtts import gTTS
import os


def distance(point1, point2):
    return math.sqrt((abs(point1[0] - point2[0]) ** 2) + (abs(point1[1] - point2[1]) ** 2))


UbidotsApi = ApiClient(token="BBFF-Yqq576MyOD7OOWMIsCPto8CRqp64ZO")

# Create face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('FaceModel2.yml')
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

LED_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera can not open")
    exit()

FacePoint = [(0, 0) for i in range(70)]
NowTime = time.time()
NowTime2 = time.time()

LoadingText = [".", "..", "..."]
LoadingTextIdx = 0

fps = 0
Num = 0
checkTired = 0.0
checkMouthOpen = 0.0

name = {
    '1': 'Leo',
    '2': 'Chen'
}

VariableArr = {
    '1': '63c1890182d6f4000c6a555f',
    '2': '63c249eab019de000cf16ef3'
}

UserName = '???'
UserVariable = ''

checkVerify = False
CheckingText = 'Checking...'
checkCount = 0.0

count = 0

while True:
    fps += 1

    # Verification check
    if time.time() - NowTime > 8:
        if checkVerify:
            break
        else:
            CheckingText = 'Verify failed'

    # Loading text update
    if time.time() - NowTime2 > 1:
        NowTime2 = time.time()
        LoadingTextIdx = (LoadingTextIdx + 1) % 3

    _, frame = cap.read()
    frame = cv2.resize(frame, (400, 300))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)  # Detect face

    # User verification
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
        idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])  # Get confidence

        if confidence < 60:
            checkCount += 1
            if time.time() - NowTime > 5 and not checkVerify:
                VerifyConfidence = checkCount / fps
                if VerifyConfidence > 0.5:
                    checkVerify = True
                    text = name[str(idnum)]
                    UserName = name[str(idnum)]
                    UserVariable = VariableArr[str(idnum)]
                    CheckingText = 'Verify pass!'
                else:
                    text = LoadingText[LoadingTextIdx]
            elif checkVerify:
                text = name[str(idnum)]
            else:
                text = LoadingText[LoadingTextIdx]
        else:
            text = LoadingText[LoadingTextIdx]

        # Display name
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display additional information on frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    count = (count + 1) % 25
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 400, 300)
    if count % 25 > 1:
        cv2.putText(frame, CheckingText, (0, 25), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Image', frame)

variable = UbidotsApi.get_variable(UserVariable)

# LED and Timer settings
CheckBlink = False
CheckLedState = 0
TimerBlink = time.time()
TimerPerMinute = time.time()
TiredCountPerMinute = 0.0
fpm = 0
fps = 0

while True:
    fps += 1
    fpm += 1

    # LED Blinking Logic
    if time.time() - TimerBlink > 1:
        if CheckBlink:
            GPIO.output(LED_PIN, GPIO.HIGH if CheckLedState == 0 else GPIO.LOW)
            CheckLedState = 1 - CheckLedState
        elif CheckLedState == 1:
            GPIO.output(LED_PIN, GPIO.LOW)
            CheckLedState = 0
        TimerBlink = time.time()

    # Calculate Tired Rate per Minute
    if time.time() - TimerPerMinute > 60:
        TiredRateMinute = (TiredCountPerMinute / fpm) * 100
        variable.save_value({'value': format(TiredRateMinute, '.2f')})
        fpm, TiredCountPerMinute = 0, 0.0
        TimerPerMinute = time.time()

    # Check Tired Rate every 2 seconds
    if time.time() - NowTime > 2:
        MouthOpenRate = checkMouthOpen / fps
        TiredRate = checkTired / fps
        NowTime, fps, checkTired, checkMouthOpen = time.time(), 0, 0.0, 0.0

        # Warning the user based on detection
        if MouthOpenRate > 0.8:
            tts = gTTS(text='Warning! Detected yawning', lang='en')
            tts.save('hello_tw.mp3')
            os.system('omxplayer -o local -p hello_tw.mp3 > /dev/null 2>&1')
            time.sleep(1)
            print('Detected yawning')
        elif TiredRate > 0.8:
            CheckBlink = True
            tts = gTTS(text='Warning! Wake up', lang='en')
            tts.save('hello_tw.mp3')
            os.system('omxplayer -o local -p hello_tw.mp3 > /dev/null 2>&1')
            time.sleep(1)
            print('Warning! Wake up')
        else:
            CheckBlink = False

    _, frame = cap.read()
    frame = cv2.resize(frame, (400, 300))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Face detection and landmark plotting
    if faces:
        face = faces[0]
        landmarks = predictor(gray, face)
        _, jpeg = cv2.imencode('.jpg', gray)
        for idx, point in enumerate(landmarks.parts(), start=1):
            FacePoint[idx] = (point.x, point.y)
            if idx > 37:
                cv2.circle(gray, (point.x, point.y), 3, (0, 0, 255), -1)

    # EAR Calculation
    leftEAR, rightEAR, mouthEAR, TotalEAR = calculate_ear(FacePoint)
    eyeEARlimit = 0.22

    if TotalEAR < eyeEARlimit:
        TiredCountPerMinute += 1
        checkTired += 1
    if mouthEAR > 1.0:
        TiredCountPerMinute += 1
        checkMouthOpen += 1

    # Display information on frame
    display_frame_info(gray, TotalEAR, mouthEAR)

    # Exit on 'Esc' key press
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

