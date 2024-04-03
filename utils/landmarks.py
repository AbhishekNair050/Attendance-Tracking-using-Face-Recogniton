import cv2
import numpy as np
import os
import csv
import mediapipe as mp
import re
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def calculate_additional_features(face_landmarks):
    landmarks_array = np.array([[p.x, p.y] for p in face_landmarks.landmark])

    # Calculate additional features
    ear_distance = np.linalg.norm(
        landmarks_array[1] - landmarks_array[93]
    )  # Assuming landmarks_array[1] and landmarks_array[2] correspond to ear landmarks
    forehead_distance = np.linalg.norm(
        landmarks_array[8] - landmarks_array[10]
    )  # Assuming landmarks_array[151] is the top of the forehead, and landmarks_array[10] is the center of the head
    nose_size = np.linalg.norm(
        landmarks_array[168] - landmarks_array[19]
    )  # Assuming landmarks_array[27] and landmarks_array[33] correspond to nose landmarks

    return ear_distance, forehead_distance, nose_size


def process_dataset(dataset_directory, csv_file):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # Create a list to store all landmarks
    all_landmarks = []
    all_labels = []

    # Iterate over all the subfolders in the dataset directory
    for folder_name in os.listdir(dataset_directory):
        folder_path = os.path.join(dataset_directory, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Iterate over image files
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)

                # Check if image file
                if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image = cv2.imread(image_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Process image with Face Mesh
                    results = face_mesh.process(image_rgb)

                    # If landmarks detected
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]

                        # Calculate additional features
                        ear_distance, forehead_distance, nose_size = (
                            calculate_additional_features(face_landmarks)
                        )

                        # Get label from folder name
                        label = folder_name

                        # Extract landmarks
                        landmarks = (
                            [p.x for p in face_landmarks.landmark]
                            + [p.y for p in face_landmarks.landmark]
                            + [p.z for p in face_landmarks.landmark]
                            + [ear_distance, forehead_distance, nose_size]
                        )

                        # Store landmarks and label
                        all_landmarks.append(landmarks)
                        all_labels.append(label)

    # Perform dimensionality reduction and feature selection
    selected_features = feature_selection(landmarks, n_features=500)

    # Open the CSV file in write mode
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Write the selected features to CSV
        for i in range(len(all_landmarks)):
            row = [all_labels[i]]
            for j in selected_features:
                if j < len(all_landmarks[i]):
                    row.append(all_landmarks[i][j])
            writer.writerow(row)


def feature_selection(landmarks, n_components=1, n_features=20):
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


def read_landmarks_from_csv(file_path):
    landmarks_data = []
    labels = []

    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row if present
        for row in csv_reader:
            label = row[0]
            landmarks = [float(coord) for coord in row[1:]]

            landmarks_data.append(landmarks)
            labels.append(label)

    return landmarks_data, labels
