import tkinter as tk
from tkinter import *
import os, cv2
import shutil
import csv
import numpy as np
from PIL import ImageTk, Image
import pandas as pd
import pyttsx3

import show_attendance
import takeImage
import trainImage
import automaticAttedance

def text_to_speech(user_text):
    engine = pyttsx3.init()
    engine.say(user_text)
    engine.runAndWait()

# File paths
haarcasecade_path = r"./haarcascade_frontalface_default.xml"
trainimagelabel_path = r"./TrainingImageLabel\Trainner.yml"
trainimage_path = r"./trainimage"
studentdetail_path = r"./StudentDetails\studentdetails.csv"
attendance_path = r"./Attendance"

os.makedirs(trainimage_path, exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)
os.makedirs("StudentDetails", exist_ok=True)
os.makedirs(attendance_path, exist_ok=True)

window = Tk()
window.title("Face Recognizer")
window.geometry("1280x720")
window.configure(background="#2C3E50")


def err_screen():
    sc1 = tk.Toplevel()
    sc1.geometry("400x110")
    sc1.title("Warning!!")
    sc1.configure(background="Cyan")
    tk.Label(
        sc1, text="Enrollment & Name required!!!", fg="yellow", bg="#2C3E50", font=("times", 20, "bold")
    ).pack()
    tk.Button(
        sc1, text="OK", command=sc1.destroy, fg="yellow", bg="#2C3E50", font=("times", 20, "bold")
    ).pack()

def testVal(inStr, acttyp):
    if acttyp == "1" and not inStr.isdigit():
        return False
    return True

logo = Image.open(r"./UI_Image\0003.png").resize((50, 47), Image.Resampling.LANCZOS)
logo1 = ImageTk.PhotoImage(logo)
tk.Label(window, bg="#2C3E50", relief=RIDGE, bd=10, font=("arial", 35)).pack(fill=X)
tk.Label(window, image=logo1, bg="#2C3E50").place(x=470, y=10)
tk.Label(
    window, text="Arya's College!!", bg="#2C3E50", fg="green", font=("arial", 27)
).place(x=525, y=12)
tk.Label(
    window,
    text="Welcome to the Face Recognition Based\nAttendance Management System",
    bg="#2C3E50",
    fg="yellow",
    bd=10,
    font=("arial", 35),
).pack()


register_img = ImageTk.PhotoImage(Image.open(r"./UI_Image\register.png"))
attendance_img = ImageTk.PhotoImage(Image.open(r"./UI_Image\attendance.png"))
verify_img = ImageTk.PhotoImage(Image.open(r"./UI_Image\verifyy.png"))

tk.Label(window, image=register_img, bg="#2C3E50").place(x=100, y=270)
tk.Label(window, image=attendance_img, bg="#2C3E50").place(x=980, y=270)
tk.Label(window, image=verify_img, bg="#2C3E50").place(x=600, y=270)

def TakeImageUI():
    ImageUI = Toplevel()
    ImageUI.title("Take Student Image")
    ImageUI.geometry("780x480")
    ImageUI.configure(background="#2C3E50")
    ImageUI.resizable(0, 0)

    tk.Label(
        ImageUI, text="Register Your Face", bg="#2C3E50", fg="red", font=("arial", 32 , "bold")
    ).place(x=270, y=12)

    tk.Label(
        ImageUI, text="Enter the details", bg="#2C3E50", fg="yellow", font=("arial", 24)
    ).place(x=280, y=75)

    tk.Label(
        ImageUI,
        text="Enrollment No",
        width=10,
        height=2,
        bg="#2C3E50",
        fg="yellow",
        bd=5,
        relief=RIDGE,
        font=("times new roman", 12 , "bold"),
    ).place(x=120, y=130)
    txt1 = tk.Entry(
        ImageUI,
        width=17,
        bd=5,
        validate="key",
        bg="#2C3E50",
        fg="yellow",
        relief=RIDGE,
        font=("times", 25, "bold"),
    )
    txt1.place(x=250, y=130)
    txt1["validatecommand"] = (txt1.register(testVal), "%P", "%d")


    tk.Label(
        ImageUI,
        text="Name",
        width=10,
        height=2,
        bg="#2C3E50",
        fg="yellow",
        bd=5,
        relief=RIDGE,
        font=("times new roman", 12),
    ).place(x=120, y=200)
    txt2 = tk.Entry(
        ImageUI,
        width=17,
        bd=5,
        bg="#2C3E50",
        fg="yellow",
        relief=RIDGE,
        font=("times", 25, "bold"),
    )
    txt2.place(x=250, y=200)

    tk.Label(
        ImageUI,
        text="Notification",
        width=10,
        height=2,
        bg="#2C3E50",
        fg="yellow",
        bd=5,
        relief=RIDGE,
        font=("times new roman", 12),
    ).place(x=120, y=270)
    message = tk.Label(
        ImageUI,
        text="",
        width=32,
        height=2,
        bd=5,
        bg="#2C3E50",
        fg="yellow",
        relief=RIDGE,
        font=("times", 12, "bold"),
    )
    message.place(x=250, y=270)

    def take_image():
        l1 = txt1.get()
        l2 = txt2.get()
        takeImage.TakeImage(
            l1, l2, haarcasecade_path, trainimage_path, message, err_screen, text_to_speech
        )
        txt1.delete(0, "end")
        txt2.delete(0, "end")

    tk.Button(
        ImageUI,
        text="Take Image",
        command=take_image,
        bd=10,
        font=("times new roman", 18),
        bg="#2C3E50",
        fg="yellow",
        height=2,
        width=12,
    ).place(x=130, y=350)

    def train_image():
        trainImage.TrainImage(
            haarcasecade_path,
            trainimage_path,
            trainimagelabel_path,
            message,
            text_to_speech,
        )

    tk.Button(
        ImageUI,
        text="Train Image",
        command=train_image,
        bd=10,
        font=("times new roman", 18),
        bg="#2C3E50",
        fg="yellow",
        height=2,
        width=12,
    ).place(x=360, y=350)

tk.Button(
    window,
    text="Register a new student",
    command=TakeImageUI,
    bd=10,
    font=("times new roman", 16),
    bg="#2C3E50",
    fg="yellow",
    height=2,
    width=17,
).place(x=100, y=520)

def automatic_attendance():
    automaticAttedance.subjectChoose(text_to_speech)

tk.Button(
    window,
    text="Take Attendance",
    command=automatic_attendance,
    bd=10,
    font=("times new roman", 16),
    bg="#2C3E50",
    fg="yellow",
    height=2,
    width=17,
).place(x=600, y=520)

def view_attendance():
    show_attendance.subjectchoose(text_to_speech)

tk.Button(
    window,
    text="View Attendance",
    command=view_attendance,
    bd=10,
    font=("times new roman", 16),
    bg="#2C3E50",
    fg="yellow",
    height=2,
    width=17,
).place(x=1000, y=520)

tk.Button(
    window,
    text="EXIT",
    command=quit,
    bd=10,
    font=("times new roman", 16),
    bg="#2C3E50",
    fg="yellow",
    height=2,
    width=17,
).place(x=600, y=660)

window.mainloop()
