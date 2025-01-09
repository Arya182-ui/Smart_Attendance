import pandas as pd
from glob import glob
import os
import tkinter as tk
from tkinter import messagebox
import csv


def text_to_speech(text):
    print(f"[TTS]: {text}")


def subjectchoose(text_to_speech):
    def validate_subject(subject):
        subject_dir = f"Attendance/{subject}"
        if not os.path.exists(subject_dir):
            t = f"Subject '{subject}' not found."
            text_to_speech(t)
            return False
        return True

    def calculate_attendance():
        subject = tx.get()
        if not subject:
            t = 'Please enter the subject name.'
            text_to_speech(t)
            return

        if not validate_subject(subject):
            return

        os.chdir(f"Attendance/{subject}")

        filenames = glob(f"{subject}*.csv")
        if not filenames:
            t = f"No attendance files found for '{subject}'."
            text_to_speech(t)
            return

        try:
            df = [pd.read_csv(f) for f in filenames]
            newdf = df[0]
            for i in range(1, len(df)):
                newdf = newdf.merge(df[i], how="outer")

            newdf.fillna(0, inplace=True)
            newdf["Attendance"] = 0

            for i in range(len(newdf)):
                newdf["Attendance"].iloc[i] = str(int(round(newdf.iloc[i, 2:-1].mean() * 100))) + '%'

            newdf.to_csv("attendance.csv", index=False)

            display_attendance(newdf)

        except Exception as e:
            error_msg = f"Error processing attendance: {str(e)}"
            messagebox.showerror("Error", error_msg)
            text_to_speech(error_msg)

    def display_attendance(newdf):
        root = tk.Tk()
        root.title(f"Attendance for {tx.get()}")
        root.configure(background="#2C3E50")

        for r, row in enumerate(newdf.values.tolist()):
            for c, col in enumerate(row):
                label = tk.Label(root, text=col, width=15, height=1, fg="yellow", bg="#2C3E50", font=("times", 15), relief=tk.RIDGE)
                label.grid(row=r, column=c)

        root.mainloop()

    def Attf():
        sub = tx.get()
        if sub == "":
            t = "Please enter the subject name!"
            text_to_speech(t)
        else:
            os.startfile(f"Attendance/{sub}")

    subject = tk.Tk()
    subject.title("Subject Selection")
    subject.geometry("580x320")
    subject.resizable(0, 0)
    subject.configure(background="#2C3E50")

    titl = tk.Label(subject, bg="#2C3E50", relief=tk.RIDGE, bd=10, font=("arial", 30))
    titl.pack(fill=tk.X)

    titl = tk.Label(subject, text="Which Subject of Attendance?", bg="#2C3E50", fg="green", font=("arial", 25))
    titl.place(x=100, y=12)

    sub = tk.Label(subject, text="Enter Subject", width=10, height=2, bg="#2C3E50", fg="yellow", bd=5, relief=tk.RIDGE, font=("times new roman", 15))
    sub.place(x=50, y=100)

    tx = tk.Entry(subject, width=15, bd=5, bg="#2C3E50", fg="yellow", relief=tk.RIDGE, font=("times", 30, "bold"))
    tx.place(x=190, y=100)

    fill_a = tk.Button(subject, text="View Attendance", command=calculate_attendance, bd=7, font=("times new roman", 15), bg="#2C3E50", fg="yellow", height=2, width=12, relief=tk.RIDGE)
    fill_a.place(x=195, y=170)

    attf = tk.Button(subject, text="Check Sheets", command=Attf, bd=7, font=("times new roman", 15), bg="#2C3E50", fg="yellow", height=2, width=10, relief=tk.RIDGE)
    attf.place(x=360, y=170)

    subject.mainloop()


# # for test use :
# if __name__ == "__main__":
#     subjectchoose(text_to_speech)
