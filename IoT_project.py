from tkinter import messagebox as mb
from PIL import Image, ImageDraw, ImageFont
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import face_recognition
import cv2
import os
import pickle
import dlib
import serial

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

g_user = 'unknown'
g_cascade = 'haarcascade_frontalface_default.xml'


def put_text_pil(img: np.array, txt: str):
    im = Image.fromarray(img)

    font_size = 16
    font = ImageFont.truetype('TimesNewRoman.ttf', size=font_size)

    draw = ImageDraw.Draw(im)
    w, h = draw.textsize(txt, font=font)

    y_pos = 425
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    draw.text((int((img.shape[1] - w) / 2), y_pos), txt, fill='rgb(255, 255, 255)', font=font)

    img = np.asarray(im)

    return img


def open_the_door(name):
    ser = serial.Serial("/dev/ttyACM0", 9600)  # Open port with baud rate
    ser.write(b'name')
    print('sended')
    ser.close()


def post_login(user_name):
    global g_user
    if user_name != "unknown":
        g_user = user_name
        open_the_door(user_name)


def face_control(face_image):
    encoding_files = []
    for file in os.listdir("./users"):
        if file.endswith(".pickle"):
            encoding_files.append(os.path.join("./users", file))
    image = face_image
    detection_method = "hog"
    dic = {}
    data = {}

    for enc in encoding_files:
        with open(enc, 'rb') as f:
            dic.update(pickle.load(f))  # Update contents of file1 to the dictionary
            for key in dic:
                if key in data:
                    for val in dic[key]:
                        data[key].append(val)
                else:
                    data[key] = dic[key][:]

    image = cv2.imread(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        g_user = name
        if g_user != "unknown":
            mb.showinfo(title="Успех", message="Лицо распознано, добро пожаловать " + name + "!")
            post_login(g_user)
        else:
            mb.showinfo(title="Ошибка входа", message="Лицо не распознано, пожалуйста попробуйте еще раз либо"
                                                      " создайте новую учетную запись,"
                                                      " если не делали этого ранее")


def login_by_face():
    detector = cv2.CascadeClassifier(g_cascade)
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:
        frame = vs.read()
        orig = frame.copy()

        text = "    Вход с помощью лица\nE - войти,        Q - отмена"
        frame = put_text_pil(frame, text)
        frame = imutils.resize(frame, width=600)
        rects = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                          scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("e"):
            cv2.imwrite("face_last_login.png", orig)
            face_control("face_last_login.png")
            break
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()


login_by_face()
