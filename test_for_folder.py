import face_recognition
import joblib
import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def test_face_detection_model(test_folder, model_path, encoder_path):
    # Load the trained face encodings and labels
    face_encodings = joblib.load(model_path)
    labels = joblib.load(encoder_path)

    print("Loaded face encodings:", face_encodings)
    print("Loaded labels:", labels)

    # Create a machine learning model for face recognition
    face_model = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=True))
    face_model.fit(face_encodings, labels)

    print("Model trained.")

    # Create lists to store results
    true_labels = []
    predicted_labels = []

    # Iterate through test images
    for filename in os.listdir(test_folder):
        test_image_path = os.path.join(test_folder, filename)

        # Load the test image
        test_image = face_recognition.load_image_file(test_image_path)

        # Find face encodings in the test image
        face_encodings = face_recognition.face_encodings(test_image)

        if len(face_encodings) > 0:
            # Use the trained model to predict the label
            predictions = face_model.predict([face_encodings[0]])
            predicted_label = predictions[0]

            # Extract the true label from the filename
            true_label = filename.split('_')[0]

            # Append true and predicted labels to the lists
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

    # Print the results
    print("True labels:", true_labels)
    print("Predicted labels:", predicted_labels)

if __name__ == "__main__":
    # Specify folder containing test images
    test_images_folder = r"D:\\Computer Vision\\test_image"  # Adjust the path accordingly

    # Specify paths for the trained face detection model and encoder
    trained_model_path = r"D:\\Computer Vision\\face_detection_model.joblib"  # Adjust the path accordingly
    trained_encoder_path = r"D:\\Computer Vision\\face_encoder_labels.joblib"  # Adjust the path accordingly

    # Test the face detection model
    test_face_detection_model(test_images_folder, trained_model_path, trained_encoder_path)
