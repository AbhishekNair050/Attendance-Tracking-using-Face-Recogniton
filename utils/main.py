from utils.train import *
from utils.landmarks import *
from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np
import pickle
import mediapipe as mp
from datetime import date, datetime

present_students = set()


def process_predictions(preds, subject):
    for student_id in preds:
        if student_id not in present_students:
            subject = subject
            time = datetime.now().time()
            today = date.today()
            date1 = today.strftime("%d/%m/%Y")
            present_students.add(student_id)
            # Write attendance data to CSV file
            with open("attendance.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([student_id, "Present", date1, time, subject])


import numpy as np
from sklearn.decomposition import PCA


import numpy as np
from sklearn.decomposition import PCA


def feature_selection(landmarks, n_components=0.95, n_features=20):
    # Reshape landmarks to a 2D array
    landmarks = np.array(landmarks).reshape(-1, 1).T

    # Perform dimensionality reduction with PCA
    pca = PCA(n_components=n_components)
    landmarks_pca = pca.fit_transform(landmarks)

    # Get the loadings (coefficients) of the principal components
    loadings = pca.components_

    # Compute the importance scores based on the absolute value of the loadings
    importance_scores = np.sum(np.abs(loadings), axis=0)

    # Get the indices of the most important features
    important_features = np.argsort(importance_scores)[-n_features:]

    return important_features.flatten()


def face_recognition_function(subject):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    cap = cv2.VideoCapture(0)
    rf = pickle.load(open("utils/model.pkl", "rb"))
    subject = subject
    while True:
        ret, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get bounding box
                x_min = int(
                    min([p.x for p in face_landmarks.landmark]) * frame.shape[1]
                )
                y_min = int(
                    min([p.y for p in face_landmarks.landmark]) * frame.shape[0]
                )
                x_max = int(
                    max([p.x for p in face_landmarks.landmark]) * frame.shape[1]
                )
                y_max = int(
                    max([p.y for p in face_landmarks.landmark]) * frame.shape[0]
                )

                landmarks = []
                for landmark in results.multi_face_landmarks[0].landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Calculate additional features
                ear_distance = np.linalg.norm(np.array([landmarks[1] - landmarks[93]]))
                forehead_distance = np.linalg.norm(
                    np.array([landmarks[8] - landmarks[10]])
                )
                nose_size = np.linalg.norm(np.array([landmarks[168] - landmarks[19]]))

                # Append additional features to landmarks
                landmarks.extend([ear_distance, forehead_distance, nose_size])
                all_landmarks = []
                all_landmarks.append(landmarks)
                if all_landmarks:
                    selected_features = feature_selection(landmarks, n_features=500)
                    selected_features = np.array(all_landmarks)[:, selected_features]
                    print("Number of features in input:", len(selected_features))
                    preds = rf.predict(selected_features)
                    process_predictions(preds, subject)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    cv2.putText(
                        frame,
                        str(preds[0]),
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Live Attendance Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            ret, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        else:
            continue

    # Release resources and close the window
    cv2.destroyAllWindows()
