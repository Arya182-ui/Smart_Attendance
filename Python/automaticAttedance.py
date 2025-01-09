import os
import cv2
import pandas as pd
import time
import datetime
import tkinter as tk
from tkinter import *
import tkinter.ttk as tkk
import tkinter.font as font
import csv



haarcasecade_path = r"haarcascade_frontalface_default.xml"
trainimagelabel_path = r"TrainingImageLabel\Trainner.yml"
trainimage_path = r"trainimage"
studentdetail_path = r"StudentDetails\studentdetails.csv"
attendance_path = r"Attendance"

if not os.path.exists(attendance_path):
    os.makedirs(attendance_path)

def subjectChoose(text_to_speech):
    def FillAttendance():
        sub = tx.get()  
        print(f"Subject entered: {sub}")  
        now = time.time()
        future = now + 20
        print(now)
        print(future)

        if sub == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
            print("No subject entered!")  
        else:
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                try:
                    recognizer.read(trainimagelabel_path)
                except:
                    e = "Model not found, please train model"
                    Notifica.configure(
                        text=e,
                        bg="#2C3E50",
                        fg="yellow",
                        width=33,
                        font=("times", 15, "bold"),
                    )
                    Notifica.place(x=20, y=250)
                    text_to_speech(e)
                    return

                facecasCade = cv2.CascadeClassifier(haarcasecade_path)
                df = pd.read_csv(studentdetail_path)
                cam = cv2.VideoCapture(0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                col_names = ["Enrollment", "Name"]
                attendance = pd.DataFrame(columns=col_names)

                while True:
                    ___, im = cam.read()
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = facecasCade.detectMultiScale(gray, 1.2, 5)

                    for (x, y, w, h) in faces:
                        global Id
                        Id, conf = recognizer.predict(gray[y : y + h, x : x + w])
                        if conf < 70:
                            global Subject
                            global aa
                            global date
                            global timeStamp
                            Subject = tx.get()
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                            aa = df.loc[df["Enrollment"] == Id, "Name"].values[0]
                            tt = str(Id) + "-" + aa
                            attendance.loc[len(attendance)] = [Id, aa]
                        else:
                            Id = "Unknown"
                            tt = str(Id)
                            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
                            cv2.putText(im, str(tt), (x + h, y), font, 1, (0, 25, 255), 4)

                    if time.time() > future:
                        break

                    attendance = attendance.drop_duplicates(["Enrollment"], keep="first")
                    cv2.imshow("Filling Attendance...", im)

                    key = cv2.waitKey(30) & 0xFF
                    if key == 27:
                        break

                ts = time.time()

                print(f"Attendance Data:\n{attendance}")  

                attendance[date] = 1
                attendance = attendance.drop_duplicates(["Enrollment"], keep="first")
                
                Hour, Minute, Second = timeStamp.split(":")

                subject_directory = f"{attendance_path}/{sub}"
                if not os.path.exists(subject_directory):
                    os.makedirs(subject_directory)
                    print(f"Created directory for subject: {subject_directory}")  

                fileName = f"{subject_directory}/{sub}_{date}_{Hour}-{Minute}-{Second}.csv"
                print(f"Saving attendance to: {fileName}")  

                attendance.to_csv(fileName, index=False)

                m = f"Attendance Filled Successfully for {sub}"
                Notifica.configure(
                    text=m,
                    bg="#2C3E50",
                    fg="yellow",
                    width=33,
                    relief=RIDGE,
                    bd=5,
                    font=("times", 15, "bold"),
                )
                text_to_speech(m)
                Notifica.place(x=20, y=250)

                cam.release()
                cv2.destroyAllWindows()

                root = tk.Tk()
                root.title(f"Attendance of {sub}")
                root.configure(background="#2C3E50")
                with open(fileName, newline="") as file:
                    reader = csv.reader(file)
                    r = 0
                    for col in reader:
                        c = 0
                        for row in col:
                            label = tk.Label(
                                root,
                                width=10,
                                height=1,
                                fg="yellow",
                                font=("times", 15, " bold "),
                                bg="#2C3E50",
                                text=row,
                                relief=tk.RIDGE,
                            )
                            label.grid(row=r, column=c)
                            c += 1
                        r += 1
                root.mainloop()

            except Exception as e:
                f = "No Face found for attendance"
                text_to_speech(f)
                cv2.destroyAllWindows()
                print(str(e)) 
    subject = Tk()
    subject.title("Subject Selection")
    subject.geometry("580x320")
    subject.resizable(0, 0)
    subject.configure(background="#2C3E50")

    titl = tk.Label(subject, bg="#2C3E50", relief=RIDGE, bd=10, font=("arial", 30))
    titl.pack(fill=X)

    titl = tk.Label(
        subject,
        text="Enter the Subject Name",
        bg="#2C3E50",
        fg="green",
        font=("arial", 25),
    )
    titl.place(x=160, y=12)

    Notifica = tk.Label(
        subject,
        text="Attendance filled Successfully",
        bg="yellow",
        fg="#2C3E50",
        width=33,
        height=2,
        font=("times", 15, "bold"),
    )

    def Attf():
        sub = tx.get()
        if sub == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
        else:
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            Hour, Minute, Second = timeStamp.split(":")
            file_name = f"C:/Users/techg/OneDrive/Documents/Desktop/Project/Attendance/{sub}/{sub}_{date}_{Hour}-{Minute}-{Second}.csv"
            
            if os.path.exists(file_name):  
                root = tk.Tk()
                root.title(f"Attendance of {sub}")
                root.configure(background="#2C3E50")

                with open(file_name, newline="") as file:
                    reader = csv.reader(file)
                    r = 0
                    for col in reader:
                        c = 0
                        for row in col:
                            label = tk.Label(
                                root,
                                width=10,
                                height=1,
                                fg="yellow",
                                font=("times", 15, " bold "),
                                bg="#2C3E50",
                                text=row,
                                relief=tk.RIDGE,
                            )
                            label.grid(row=r, column=c)
                            c += 1
                        r += 1
                
                root.mainloop()
            else:
                error_message = f"Attendance sheet for {sub} not found!"
                text_to_speech(error_message)
                print(error_message)

    attf = tk.Button(
        subject,
        text="Check Sheets",
        command=Attf,
        bd=7,
        font=("times new roman", 15),
        bg="#2C3E50",
        fg="yellow",
        height=2,
        width=10,
        relief=RIDGE,
    )
    attf.place(x=360, y=170)

    sub = tk.Label(
        subject,
        text="Enter Subject",
        width=10,
        height=2,
        bg="#2C3E50",
        fg="yellow",
        bd=5,
        relief=RIDGE,
        font=("times new roman", 15),
    )
    sub.place(x=50, y=100)

    tx = tk.Entry(
        subject,
        width=15,
        bd=5,
        bg="#2C3E50",
        fg="yellow",
        relief=RIDGE,
        font=("times", 30, "bold"),
    )
    tx.place(x=190, y=100)

    fill_a = tk.Button(
        subject,
        text="Fill Attendance",
        command=FillAttendance,
        bd=7,
        font=("times new roman", 15),
        bg="#2C3E50",
        fg="yellow",
        height=2,
        width=12,
        relief=RIDGE,
    )
    fill_a.place(x=195, y=170)

    subject.mainloop()
