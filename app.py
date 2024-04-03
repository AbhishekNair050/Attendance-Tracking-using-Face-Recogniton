import time
from utils.main import face_recognition_function
from sklearn.ensemble import RandomForestClassifier
import csv
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
import os
import numpy as np
import cv2
import random
import warnings
import json

warnings.filterwarnings("ignore")
# from utils.sip import process_dataset, read_landmarks_from_csv

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("email")
        password = request.form.get("password")
        if user and password:
            with open("utils/data.json", "r") as file:
                data = json.load(file)
            if user in data and data[user]["password"] == password:
                return redirect(url_for("homepage"))
        return render_template("login.html")
    elif request.method == "GET":
        return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        user = request.form.get("email")
        password = request.form.get("password")
        name = request.form.get("fullname")
        roll = request.form.get("roll-no")
        course = request.form.get("course")
        if user and password and name:
            with open(r"D:\College\Project\ML_SIP\utils\data.json", "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = {}
            if user in data:
                return "User already exists", 400
            else:
                data[user] = {
                    "password": password,
                    "name": name,
                    "roll": roll,
                    "course": course,
                }
                with open(r"D:\College\Project\ML_SIP\utils\data.json", "w") as file:
                    json.dump(data, file)
                return "User created successfully", 201
    return render_template("signup.html")


def image_capture(name, image_data):
    # Capture the images
    face_cascade = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")
    # Convert base64 image data to numpy array
    frame = image_data

    count = random.randint(1, 500)

    # Create the 'Dataset' directory if it doesn't exist
    dataset_dir = "Dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Create the subdirectory inside 'Dataset' for the user's name
    user_dir = os.path.join(dataset_dir, name)
    os.makedirs(user_dir, exist_ok=True)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        image_path = os.path.join(user_dir, f"{name}_{count}.jpg")  # Adjust filename
        cv2.imwrite(
            image_path, frame[y : y + h, x : x + w]
        )  # Use the original frame instead of gray

    return count


@app.route("/capture", methods=["POST"])
def capture():
    frame_file = request.files["frame"]
    frame_data = bytearray(frame_file.read())
    frame_np = np.asarray(frame_data, dtype=np.uint8)
    frame_np = np.array(frame_np)
    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    try:
        name = request.headers.get("name")
        print(name)
        count = image_capture(name, frame)
        return jsonify({"message": f"{count} images captured for {name}"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route("/home")
def homepage():
    absent = 0
    present = 0
    attendance_data = []
    with open("attendance.csv", "r") as file:
        reader = csv.reader(file)
        data = list(reader)
        for row in data:
            attendance_data.append(row)
            if "Absent" in row:
                absent = absent + 1
            elif "Present" in row:
                present = present + 1
    occupied_datas = [
        {"name": "Absent", "value": absent},
        {"name": "Present", "value": present},
    ]
    chart3 = {
        "chart3": {
            "type": "pie",
            "data": occupied_datas,
            "container": "container3",
        }
    }
    print(attendance_data)
    return render_template(
        "home.html",
        chart3=json.dumps(chart3),
        occupied_datas=json.dumps(occupied_datas),
        attendance_data=attendance_data,
    )


@app.route("/student")
def student():
    formatted_data = []
    with open("utils/data.json", "r") as file:
        data = json.load(file)
        for key, value in data.items():
            formatted_data.append([value["name"], key, value["roll"], value["course"]])
    print(formatted_data)
    return render_template("student.html", data=formatted_data)


@app.route("/report")
def report():
    # all unique name in attendance.csv, all their details and another row with total percentage present
    # Load initial attendance data from CSV
    with open("attendance.csv", "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    student_data = {}
    for row in data:
        if row[0] not in student_data:
            student_data[row[0]] = {"name": row[0], "present": 0, "absent": 0}
        if "Present" in row:
            student_data[row[0]]["present"] += 1
        elif "Absent" in row:
            student_data[row[0]]["absent"] += 1
    for student in student_data.values():
        total = student["present"] + student["absent"]
        student["percentage"] = (student["present"] / total) * 100
    student_data = list(student_data.values())
    attendance_list = []

    for entry in student_data:
        attendance_list.append(
            [entry["name"], entry["present"], entry["absent"], entry["percentage"]]
        )
    return render_template("report.html", data=attendance_list)


@app.route("/attendance")
def attendance():
    # Load initial attendance data from CSV
    current_attendance = load_attendance_data()
    return render_template("attendance.html", current_attendance=current_attendance)


def load_attendance_data():
    # Read the latest attendance data from the CSV file
    current_attendance = []
    with open("attendance.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            current_attendance.append(row)
    return current_attendance


prev_attendance = load_attendance_data()


@app.route("/attendance_data")
def attendance_data():
    global prev_attendance
    current_attendance = []
    with open("attendance.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row not in prev_attendance:
                current_attendance.append(row)
    return jsonify(current_attendance)


def load_attendance_data():
    # Read the latest attendance data from the CSV file
    current_attendance = []
    with open("attendance.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            current_attendance.append(row)
    return current_attendance


@app.route("/video_feed", methods=["POST"])
def video_feed():
    subject = request.json.get("subject")
    return Response(
        face_recognition_function(subject),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/add_student")
def add_student():
    return render_template("add_student.html")


if __name__ == "__main__":
    app.run(debug=True)
