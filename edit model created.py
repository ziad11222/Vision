import face_recognition
import joblib
import os

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

    # Create and train a face recognition model using the collected face encodings and labels
    print("Training face recognition model...")
    # Note: There is no direct function in face_recognition for training a model with labels.
    # You might want to use a machine learning library like scikit-learn for this purpose.

    # Save the face model and encoder to files
    print("Saving face recognition model and encoder to files...")
    joblib.dump(face_encodings, output_model_path)
    joblib.dump(labels, output_encoder_path)

    print("Model creation completed.")

if __name__ == "__main__":
    # Specify input folder containing face recognition frames
    input_folder_faces = r"D:\\Computer Vision\\frames_faces"  # Adjust the path accordingly
    # Specify output paths for the face detection model and encoder
    output_model_path = r"D:\\Computer Vision\\manyfacesmodel.joblib"  # Adjust the path accordingly
    output_encoder_path = r"D:\\Computer Vision\\manyfacesencoder.joblib"  # Adjust the path accordingly

    # Train the face detection model and save it to files
    train_face_detection_model(input_folder_faces, output_model_path, output_encoder_path)
