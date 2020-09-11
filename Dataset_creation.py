import tkinter as tk
from tkinter import messagebox as mb
from tkinter import StringVar
from PIL import Image, ImageDraw, ImageFont
from imutils.video import VideoStream
import numpy as np
import shutil
import imutils
import time
from imutils import paths
import face_recognition
import cv2
import os
import pickle
import dlib

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

g_user = 'unknown'
g_cascade = 'haarcascade_frontalface_default.xml'


class Main(tk.Frame):
    global g_user

    def __init__(self, root):
        super().__init__(root)
        self.init_main()

    def init_login(self):
        global g_cascade

    def init_main(self):
        global g_user

        def post_login(user_name):
            global g_user
            if user_name != "unknown":
                g_user = user_name
                btn_avatar_dialog.config(state=tk.NORMAL)
                btn_avatar_dialog.config(text=g_user)
                label_authorized.config(text="Проверка пройдена!")
                btn_new.place_forget()
                btn_login.place_forget()
            else:
                btn_avatar_dialog.config(state=tk.DISABLED)
                btn_avatar_dialog.config(text="  Учетная запись ")
                label_authorized.config(text="Вы не авторизованы!")
                btn_new.place()
                btn_login.place()

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

                text = "    Проверка лица\nE - проверить,        Q - отмена"
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

        toolbar = tk.Frame(bg='#c1beef', bd=2)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.avatar_img = tk.PhotoImage(file='avatar.png')
        btn_avatar_dialog = tk.Button(toolbar, text='   Учетная запись ', command=lambda: login_by_face(),
                                      bg='#c1beef', bd=0, compound=tk.TOP, image=self.avatar_img)
        btn_avatar_dialog.pack(side=tk.RIGHT)

        label_authorized = tk.Label(root, text='Проверка не пройдена', bg='#c1beef')
        label_authorized.configure(font=("Times New Roman", 14, "italic"))
        label_authorized.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        root.login_img = tk.PhotoImage(file='login.png')
        btn_login = tk.Button(root, text='Проверить лицо', font=("Times New Roman", 10, "italic"),
                              command=lambda: login_by_face(), bg='#e6e6fa', bd=0,
                              compound=tk.TOP, image=root.login_img)
        btn_login.place(relx=0.53, rely=0.55, anchor=tk.CENTER)

        btn_new = tk.Button(root, text='Создать учетную запись ', font=("Times New Roman", 9, "italic"),
                            command=self.new_dialog, bg='#e6e6fa', bd=0, compound=tk.TOP)
        btn_new.place(relx=0.53, rely=0.75, anchor=tk.CENTER)

        if g_user != "unknown":
            btn_avatar_dialog.config(state=tk.NORMAL)
            btn_avatar_dialog.config(text=g_user)
            label_authorized.config(text="")
            btn_new.place_forget()
            btn_login.place_forget()
        else:
            btn_avatar_dialog.config(state=tk.DISABLED)
            btn_avatar_dialog.config(text="  Учетная запись ")
            label_authorized.config(text="Проверка не пройдена")
            btn_new.place()
            btn_login.place()

    def new_dialog(self):
        New_user()


class New_user(tk.Toplevel):
    def __init__(self):
        super().__init__(root)
        self.init_login()

    def init_login(self):
        global g_cascade

        def encode_faces(dataset_path, detection_method):
            dataset_path = dataset_path.replace("/", "\\")
            name = ''

            imagePaths = list(paths.list_images(dataset_path))
            knownEncodings = []
            knownNames = []

            for (i, imagePath) in enumerate(imagePaths):
                name = imagePath.split(os.path.sep)[1]

                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                boxes = face_recognition.face_locations(rgb, model=detection_method)
                encodings = face_recognition.face_encodings(rgb, boxes)
                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(name)

            data = {"encodings": knownEncodings, "names": knownNames}
            f = open("./users/" + name + ".pickle", "wb")
            pickle.dump(data, f, protocol=None)
            f.close()
            mb.showinfo(title="Обучение", message="Ваше лицо успешно запомнено!\nТеперь Вы можете проверить "
                                                  "распознавание лица в главном окне.")
            self.destroy()

        def dataset_creation(user_name):
            try:
                os.mkdir(user_name)
            except OSError:
                mb.showinfo(title="Ошибка", message="Директория " + user_name + " не может быть создана")
            else:
                detector = cv2.CascadeClassifier(g_cascade)
                vs = VideoStream(src=0).start()
                time.sleep(1.0)
                total = 0

                while True:
                    frame = vs.read()
                    orig = frame.copy()

                    text = "        Фото сделано: " + str(sum([len(files) for r, d, files in os.walk(user_name)])) + \
                           "\nS - сохранить,        Q - выход"
                    frame = put_text_pil(frame, text)
                    frame = imutils.resize(frame, width=600)
                    rects = detector.detectMultiScale(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    for (x, y, w, h) in rects:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.imshow("Webcam", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):
                        p = os.path.sep.join([user_name, "{}.png".format(str(total).zfill(5))])
                        success, im_buf_arr = cv2.imencode(p, orig)
                        if success:
                            im_buf_arr.tofile(p)

                        cv2.imwrite(p, orig)
                        total += 1
                    elif sum([len(files) for r, d, files in os.walk(user_name)]) == 10:
                        cv2.destroyAllWindows()
                        vs.stop()
                        mb.showwarning(title="Внимание!", message="Далее начнется процесс обучения,"
                                                                  " который может занять от нескольких минут времени. "
                                                                  "Пожалуйста, не завершайте работу программы до "
                                                                  "окончания обучения, Спасибо!")
                        encode_faces(user_name, "cnn")
                        break
                    elif key == ord("q"):
                        shutil.rmtree(user_name, ignore_errors=True)
                        break
                cv2.destroyAllWindows()
                vs.stop()

        def callback(name):
            if os.path.isdir("dataset/" + name.get()):
                user_entry.config(bg='red')
                btn_login.config(state=tk.DISABLED)
            else:
                user_entry.config(bg='green')
                btn_login.config(state=tk.NORMAL)

        self.title('Создание новой записи')
        self.geometry('600x400+400+300')
        self.resizable(False, False)

        label_instruction = tk.Label(self, text='Введите логин (на латинице), после чего нажмите на значок '
                                                'камеры. Сделайте фото путем нажатия клавиши "S", пока не будет '
                                                'сделано 10 фото (при съемке не должно быть посторонних определяемых '
                                                'лиц, "Q" для отмены съемки).', bg='#c1beef')
        label_instruction.configure(font=("Times New Roman", 14), wraplength=500)
        label_instruction.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

        sv = StringVar()
        sv.trace("w", lambda name, index, mode, sv=sv: callback(sv))
        user_entry = tk.Entry(self, textvariable=sv, width=50, font=("Times New Roman", 14), bd=1)
        user_entry.place(relx=0.5, rely=0.35, anchor=tk.CENTER)

        self.camera_img = tk.PhotoImage(file='camera.png')
        btn_login = tk.Button(self, font=("Times New Roman", 10, "italic"),
                              command=lambda: dataset_creation("dataset/" + sv.get()), bg='#e6e6fa', bd=0,
                              compound=tk.TOP, image=self.camera_img)
        btn_login.place(relx=0.5, rely=0.65, anchor=tk.CENTER)
        btn_login.config(state=tk.DISABLED)

        self.configure(background='#e6e6fa')
        self.grab_set()
        self.focus_set()


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


if __name__ == "__main__":
    root = tk.Tk()
    app = Main(root)
    app.pack()
    root.title("Создание записи")
    root.geometry("650x450+300+200")
    root.resizable(False, False)
    root.configure(background='#e6e6fa')
    root.mainloop()
