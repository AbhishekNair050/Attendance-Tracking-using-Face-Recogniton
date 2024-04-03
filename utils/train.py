import cv2
import numpy as np
import os


def image_capture():
    # capture the images
    face_cascade = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    count = 0
    name = input("Enter the name of the user: ")

    # Create the 'Dataset' directory if it doesn't exist
    dataset_dir = "Dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Create the subdirectory inside 'Dataset' for the user's name
    user_dir = os.path.join(dataset_dir, name)
    os.makedirs(user_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            image_path = os.path.join(user_dir, f"{name}{count}.jpg")
            cv2.imwrite(image_path, gray[y : y + h, x : x + w])
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if count >= 100:
            break
    cap.release()
    cv2.destroyAllWindows()
