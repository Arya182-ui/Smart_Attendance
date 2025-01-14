# Attendance Management System Using Face Recognition 🎓📸

An advanced Python-based Attendance Management System that leverages facial recognition technology for marking attendance. This system provides both automated and manual options, making it versatile and user-friendly. Built for educational and organizational purposes, it ensures efficiency and accuracy in attendance tracking.

## Features

### Automated Attendance

Uses LBPH (Local Binary Patterns Histogram) for facial recognition.

Detects and recognizes faces using the Haar Cascade algorithm.

Stores attendance records automatically in CSV format.

### Manual Attendance

User-friendly GUI for manual entry of attendance data.

Allows adding subjects and student details with ease.

Exports data as CSV files for further analysis.

### Data Management

Organizes training images and attendance records in a structured folder hierarchy.

Stores trained models for efficient face recognition.

### 📂 Project Structure

Attendance-Management-System/
│
├── main.py                     # Main application script
├── takemanually.py             # Manual attendance script
|-- show_attendance.py          # Manual Script for show attendace 
|__ takeimage.py                # take 50 image by a single click
|__ trainimage.py               # train images before train model
├── train_model.py              # Face recognition and training script
├── check face in image.py      # Helper functions for validation image
├── haarcascade_frontalface_default.xml  # Haar Cascade file for face detection
├── TrainingImage/              # Folder for storing training images
├── TrainingImageLabel/         # Folder for saving trained model
├── Attendance/                 # Folder for storing attendance CSV files
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation

## 🔧 Installation

Prerequisites 

Python 3.8+ (Ensure Python and pip are installed)

A working webcam (for automated attendance)


#### Step 1: Clone the Repository

``` bash 
git clone https://github.com/<your-username>/Attendance-Management-System.git
cd Attendance-Management-System 
```

#### Step 2: Install Dependencies

Install required Python packages using:

```bash
 pip install -r requirements.txt
```

#### Step 3: Configure Paths

Update the Paths acording Your System file to match your directory structure:

```python 
{
    "icon_path": "AMS.ico",
    "output_folder": "Attendance",
    "haar_cascade_path": "haarcascade_frontalface_default.xml"
}

```

#### Step 4: Prepare Training Data

Add training images in the TrainingImage folder.

Use the naming format: Name_ID.jpg (e.g., John_001.jpg).

## 🚀 Usage

Running the Application

Launch the main application:

```bash 
python main.py
```

##### **Choose one of the two modes:**

***Manual Attendance: Fill attendance via GUI.***

***Automated Attendance: Use face recognition to mark attendance.***

**Training the Model**

***To train the face recognition model:***

```bash
python attendance_recognition.py
```

**This script will:**

Load images from the TrainingImage folder.

Train the recognizer and save the model in the TrainingImageLabel folder.

Attendance Records

Automated and manual attendance records are saved as CSV files in the Attendance folder.

The filename format is: Subject_Date_Time.csv.

### 🔍 Features in Detail

**Automated Attendance**

**Face Detection: Detects faces using Haar Cascade.**

**Face Recognition: Matches faces with trained data using the LBPH algorithm.**

**Data Storage: Automatically marks attendance in real time.**

**Manual Attendance**

**Subject Entry:** Allows users to input subject names.

**Student Details:** Supports adding enrollment numbers and names.

**Data Export:** Exports attendance into easily accessible CSV files.

## CSV Format

| Enrollment Number | Name     | Date       | Attendance |
|-------------------|----------|------------|------------|
| 001               | Ayush    | 2025-01-05 | Present    |
| 002               | Subodh   | 2025-01-05 | Present    |



## **🖼️ Screenshots**

1. Automated Attendance

2. Manual Attendance



🛡️ Security Considerations

Ensure training data is stored securely.

Use a password-protected folder for sensitive student data if required.

## **📋 Dependencies**

All dependencies are listed in requirements.txt. Key libraries include:

**OpenCV:** For computer vision tasks.

**NumPy:** For data manipulation.

**Pillow:** For image processing.

**pandas:** For handling CSV data.

**tkinter:** For GUI development.

### Install them using:

```bash
pip install -r requirements.txt
```

### 🧱 Contributing

Fork this repository.

**Create a new branch for your feature:**

```bash 
git checkout -b feature-name

```

**Commit your changes:**

```bash 
git commit -m "Add feature description"
```

**Push to your branch:**

```bash 
git push origin feature-name
```

**Open a pull request.**


## **📄 License**

This project is licensed under the MIT License. See LICENSE for details.

📧 Contact

For any questions or suggestions, feel free to reach out:

Name: Ayush Gangwar

Email: arya119000@gmail.com
GitHub: https://github.com/Arya182-ui

## ☕ Support Me

If My projectshelps you  so please support me ? You can show your support by buying me a coffee! Your contributions motivate me to keep improving and building more awesome projects. 💻❤  

[![Buy Me a Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://arya182-ui.github.io/buy-a-coffee-for-me/)

