import cv2
from landmarks import *
import pickle
import warnings

warnings.filterwarnings("ignore")


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


mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()
cap = cv2.VideoCapture(0)
rf = pickle.load(open("utils/model.pkl", "rb"))
while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x_min = int(min([p.x for p in face_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([p.y for p in face_landmarks.landmark]) * frame.shape[0])
            x_max = int(max([p.x for p in face_landmarks.landmark]) * frame.shape[1])
            y_max = int(max([p.y for p in face_landmarks.landmark]) * frame.shape[0])
            landmarks = []
            for landmark in results.multi_face_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            ear_distance = np.linalg.norm(np.array([landmarks[1] - landmarks[93]]))
            forehead_distance = np.linalg.norm(np.array([landmarks[8] - landmarks[10]]))
            nose_size = np.linalg.norm(np.array([landmarks[168] - landmarks[19]]))
            landmarks = landmarks.extend([ear_distance, forehead_distance, nose_size])
            try:
                selected_features = feature_selection(landmarks, n_features=50)
                prediction = rf.predict(selected_features)
                print(prediction)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    prediction[0],
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )
                cv2.imshow("Frame", frame)
                cv2.waitKey(0)
            except:
                pass
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
