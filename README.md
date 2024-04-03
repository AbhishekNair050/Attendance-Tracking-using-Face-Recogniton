# Attendance Tracking System Using Facial Landmarks

This project aims to develop an intelligent attendance tracking system that leverages facial recognition and facial landmarks to accurately identify and record attendance. The system is designed to be user-friendly, efficient, and scalable, making it suitable for various applications such as educational institutions, workplaces, and events.

## Project Structure

The project consists of the following files:

1. **sip.py**: This file contains image processing techniques used to preprocess the input images, ensuring optimal quality for facial landmark detection and recognition.

2. **ml.py**: This file handles the machine learning aspect of the project. It extracts facial landmarks from the preprocessed images, processes the landmark data, and trains the Random Forest model for attendance tracking.

3. **train.py**: This file contains necessary functions for image capture.

4. **landmarks.py**: This file contains functions to extract facial landmarks from images using the MediaPipe library.

5. **image_capture.py**: This file is used to capture images of students or individuals for attendance tracking. When executed, it prompts the user to enter their name and captures their image, which is stored in the `Dataset` directory.

6. **app.py**: This file contains the Flask application that serves as the web interface for the attendance tracking system. It handles user interactions, displays the captured images, and integrates with the machine learning model for attendance prediction and recording.

7. **cross_val.py**: This file implements cross-validation techniques to evaluate the performance of the machine learning model and fine-tune its hyperparameters.

## Dataset

The `Dataset` directory is used to store the captured images of students or individuals for attendance tracking. Each student's or individual's images are stored in a separate directory named after their corresponding name or ID.

## Usage

1. **Image Capture**:
   - Run `image_capture.py` to capture images of students or individuals.
   - Enter the name or ID of the student/individual when prompted.
   - Captured images will be stored in the `Dataset` directory, organized by name or ID.

2. **Image Preprocessing and Landmark Extraction**:
   - Run `sip.py` to preprocess the captured images and prepare them for landmark extraction.

3. **Model Training**:
   - Run `ml.py` to extract and process the facial landmark data and train the Random Forest model.

4. **Cross-Validation**:
   - Run `cross_val.py` to perform cross-validation on the trained model and evaluate its performance.
   - Fine-tune the model's hyperparameters based on the cross-validation results for optimal performance.

5. **Web Interface**:
   - Run `app.py` to start the Flask web application.
   - Access the web interface through your preferred web browser.
   - Use the web interface to capture images, view the captured images, and initiate attendance tracking using the trained model.

## Dependencies

- Python 3.x
- Flask
- OpenCV
- MediaPipe
- scikit-learn
- NumPy
- Pandas

Make sure to install the required dependencies before running the project.

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgments

We would like to acknowledge the contributors and developers of the following libraries and tools:

- MediaPipe: https://google.github.io/mediapipe/
- Flask: https://flask.palletsprojects.com/
- OpenCV: https://opencv.org/
- scikit-learn: https://scikit-learn.org/

Their work has been instrumental in the development of this attendance tracking system.
