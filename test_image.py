import face_recognition
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_face_detection_model(input_folder, output_model_path, output_encoder_path):
    # Create a list to store face encodings and corresponding labels
    face_encodings = []
    labels = []

    # Iterate through subfolders (each representing a video)
    for video_folder in os.listdir(input_folder):
        video_folder_path = os.path.join(input_folder, video_folder)

        if os.path.isdir(video_folder_path):
            print(f"Processing video folder: {video_folder}")

            # Iterate through frames in the video's subfolder
            for filename in os.listdir(video_folder_path):
                frame_path = os.path.join(video_folder_path, filename)

                # Load the frame
                image = face_recognition.load_image_file(frame_path)

                # Find face encodings in the image
                face_encodings_in_frame = face_recognition.face_encodings(image)

                # Append all face encodings and labels (video folder name) to the lists
                for face_encoding in face_encodings_in_frame:
                    face_encodings.append(face_encoding)
                    labels.append(video_folder)

    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(face_encodings, encoded_labels, test_size=0.2, random_state=42)

    # Create and train a SVM classifier
    face_classifier = SVC(kernel='linear', probability=True)
    face_classifier.fit(X_train, y_train)

    # Save the face model and encoder to files
    print("Saving face recognition model and encoder to files...")
    joblib.dump(face_classifier, output_model_path)
    joblib.dump(label_encoder, output_encoder_path)

    print("Model creation completed.")

def test_face_detection_model(image_path, face_classifier, label_encoder):
    # Load the test image
    test_image = face_recognition.load_image_file(image_path)

    # Find face encodings in the test image
    face_encodings_in_image = face_recognition.face_encodings(test_image)

    # If faces are found, predict the labels using the face model
    if face_encodings_in_image:
        for face_encoding in face_encodings_in_image:
            # Predict the label for each face
            predicted_label = face_classifier.predict([face_encoding])
            
            # Convert the numerical label back to the original label
            predicted_label_name = label_encoder.inverse_transform(predicted_label)
            
            print("Predicted Labels:", predicted_label_name)

if __name__ == "__main__":
    # Specify input folder containing face recognition frames
    input_folder_faces = r"D:\\Computer Vision\\frames_faces"  # Adjust the path accordingly
    # Specify output paths for the face detection model and encoder
    output_model_path = r"D:\\Computer Vision\\manyfacesmodel.joblib"  # Adjust the path accordingly
    output_encoder_path = r"D:\\Computer Vision\\manyfacesencoder.joblib"  # Adjust the path accordingly

    # Train the face detection model and save it to files
    train_face_detection_model(input_folder_faces, output_model_path, output_encoder_path)

    # Load the trained face detection model and encoder
    face_classifier = joblib.load(output_model_path)
    label_encoder = joblib.load(output_encoder_path)

    # Specify the path for the test image
    test_image_path = r"D:\\Computer Vision\\test_image\\zxc.jpeg"  # Adjust the path accordingly

    # Test the face detection model on the test image
    test_face_detection_model(test_image_path, face_classifier, label_encoder)
