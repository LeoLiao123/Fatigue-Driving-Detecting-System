import dlib
import cv2
import time
import math
from ubidots import ApiClient
import time
import speech_recognition as sr
import RPi.GPIO as GPIO
from gtts import gTTS
import os                                  x


def distance(point1, point2):
    return math.sqrt((abs(point1[0] - point2[0]) ** 2) + (abs(point1[1] - point2[1]) ** 2))


#def speech(string):
 #   engine = pyttsx3.init()
  #  engine.say(string)
   # engine.runAndWait()


UbidotsApi = ApiClient(token="BBFF-Yqq576MyOD7OOWMIsCPto8CRqp64ZO")

# create face detector
detector = dlib.get_frontal_face_detector()
# Load face point predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

recognizer = cv2.face.LBPHFaceRecognizer_create()         # create recognizer
recognizer.read('FaceModel2.yml')                         # Load pre-trained face model
cascade_path = "haarcascade_frontalface_default.xml"      # Load face trace model
face_cascade = cv2.CascadeClassifier(cascade_path)

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
#voice = pyttsx3.init()

LoadingText = []
LoadingText.append(".")
LoadingText.append("..")
LoadingText.append("...")
LoadingTextIdx=0

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
UserName='???'
UserVariable=''

checkVerify = False
CheckingText = 'Checking...'
checkCount=0.0

count = 0

while True:
   fps+=1
   if time.time() - NowTime > 8 and checkVerify == True:
       break
   elif time.time() - NowTime > 8:
       CheckingText = 'Verify failed'

   if time.time() - NowTime2 > 1:
       NowTime2=time.time()
       LoadingTextIdx=(LoadingTextIdx+1)%3

   _, frame = cap.read()
   frame=cv2.resize(frame,(400,300))
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray)  # Trace face

   # Check User
   for(x,y,w,h) in faces:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)          # Draw Rectengle
       idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w])  # Get confidence
       if confidence < 60 :
           checkCount+=1
           text = LoadingText[LoadingTextIdx]
           if  time.time() - NowTime > 5 and checkVerify==False:
               VerifyConfidence=checkCount/fps
               if VerifyConfidence>0.5:
                   checkVerify = True

                   text = name[str(idnum)]
                   UserName=name[str(idnum)]
                   UserVariable=VariableArr[str(idnum)]                                # UserName   

                   CheckingText = 'Verify pass!'
           elif checkVerify==True:
               text = name[str(idnum)]
           else:
               text = LoadingText[LoadingTextIdx]
       else:
           print(LoadingTextIdx)
           text = LoadingText[LoadingTextIdx]                                          #  ???
       # WRITE NAME
       cv2.putText(frame, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

   font = cv2.FONT_HERSHEY_SIMPLEX
   count = (count + 1) % 25
   cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
   cv2.resizeWindow("Image", 400, 300)
   if count % 25 > 1:
       cv2.putText(frame, CheckingText, (0, 25), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

   cv2.imshow('Image', frame)


variable = UbidotsApi.get_variable(UserVariable)


CheckBlink=False
CheckLedState=0;

TimerBlink=time.time()

TimerPerMinute=time.time()
TiredCountPerMinute=0.0
fpm=0

fps=0
while True:
    fps += 1
    fpm += 1

    if time.time()-TimerBlink>1 and CheckBlink==True:
        if CheckLedState == 0:
            print('get')
            GPIO.output(LED_PIN, GPIO.HIGH)
            CheckLedState=1
        else:
            GPIO.output(LED_PIN, GPIO.LOW)
            CheckLedState=0
        TimerBlink=time.time()
    elif CheckBlink==False and CheckLedState==1:
        GPIO.output(LED_PIN, GPIO.LOW)
        CheckLedState=0
        TimerBlink=time.time()
    else:
        TimerBlink=time.time()
        


    if time.time()-TimerPerMinute>60:
        TiredRateMinute=TiredCountPerMinute/fpm
        TiredRateMinute*=100


        variable.save_value({'value':format(TiredRateMinute, '.2f')})
        fpm = 0
        TiredCountPerMinute = 0.0
        TimerPerMinute=time.time()

    # Every 2 sec check Tired Rate
    if time.time() - NowTime > 2:
        
        # print(fps)
        MouthOpenRate = checkMouthOpen / fps
        TiredRate = checkTired / fps
        print(checkTired)
        print(fps)
        NowTime = time.time()
        fps = 0
        checkTired = 0.0
        checkMouthOpen = 0.0

        # Check Rate and Warring the user
        if MouthOpenRate > 0.8:
            tts = gTTS(text='Warring! Detect you are yawning', lang='en')
            tts.save('hello_tw.mp3')
            os.system('omxplayer -o local -p hello_tw.mp3 > /dev/null 2>&1')
            time.sleep(1)
            NowTime = time.time()
            print('Detect yawning')
            #t = threading.Thread(target=speech("Warring detect you are yawning"))
            #t.start()
        elif TiredRate > 0.8:
            CheckBlink=True
            tts = gTTS(text='Warring! wake up', lang='en')
            tts.save('hello_tw.mp3')
            os.system('omxplayer -o local -p hello_tw.mp3 > /dev/null 2>&1')
            time.sleep(1)
            NowTime = time.time()
            print('Warring! wake up')
            #t = threading.Thread(target=speech("Warring! wake up"))
            #t.start()
        else:
            CheckBlink=False

    _, frame = cap.read()
    frame = cv2.resize(frame, (400, 300))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = detector(gray)

    if faces:
        # Take first face
        face = faces[0]

        # Detect face feature point
        landmarks = predictor(gray, face)

        _, jpeg = cv2.imencode('.jpg', gray)
        # check every fearure point
        count = 1
        for point in landmarks.parts():
            FacePoint[count] = (point.x, point.y)
            # print(count)
            count += 1
            # take the coordinate
            x = point.x
            y = point.y

            # draw the point
            if count > 37:
                cv2.circle(gray, (x, y), 3, (0, 0, 255), -1)

    mouthEAR = 0
    leftEAR = 0
    rightEAR = 0
    TotalEAR = 0
    if distance(FacePoint[37], FacePoint[40]) != 0:
        leftEAR = (distance(FacePoint[38], FacePoint[42]) + distance(FacePoint[39], FacePoint[41])) / (
                2 * distance(FacePoint[37], FacePoint[40]))
    if distance(FacePoint[46], FacePoint[43]) != 0:
        rightEAR = (distance(FacePoint[45], FacePoint[47]) + distance(FacePoint[44], FacePoint[48])) / (
                2 * distance(FacePoint[46], FacePoint[43]))
    if distance(FacePoint[62], FacePoint[68]) != 0:
        mouthEAR = (distance(FacePoint[62], FacePoint[68]) + distance(FacePoint[64], FacePoint[66])) / (
                2 * distance(FacePoint[46], FacePoint[43]))
    TotalEAR = (leftEAR + rightEAR) / 2.0

    eyeEARlimit = 0.22

    if TotalEAR < eyeEARlimit:
        TiredCountPerMinute+=1
        checkTired += 1
    if mouthEAR > 1.0:
        TiredCountPerMinute += 1
        checkMouthOpen += 1


    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 400, 300)

    font = cv2.FONT_HERSHEY_SIMPLEX

    Num = (Num + 1) % 25
    eyeEAR = "eyesEAR: %f" % TotalEAR
    MouthEAR = "mouthEAR: %f" % mouthEAR


    if Num % 25 > 1:
        cv2.putText(gray, "Press 'Esc' to exit", (0, 25), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(gray, eyeEAR, (0, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(gray, MouthEAR, (0, 75), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow("Image", gray)

    # press Esc to exit program
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllwindows()

