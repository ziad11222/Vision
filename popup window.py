from PIL import Image, ImageTk
import face_recognition
import joblib
import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import Label, Canvas, PhotoImage, Scrollbar, VERTICAL, Y

def save_faces_from_image(image, face_locations, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_image = Image.fromarray(image[top:bottom, left:right])
        output_path = os.path.join(output_folder, f"face_{i + 1}.jpg")
        face_image.save(output_path)

def test_face_detection_model(test_folder, model_path, encoder_path, output_folder):
    face_encodings = joblib.load(model_path)
    labels = joblib.load(encoder_path)

    print("Loaded face encodings:", face_encodings)
    print("Loaded labels:", labels)

    face_model = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=True))
    face_model.fit(face_encodings, labels)

    print("Model trained.")

    true_labels = []
    predicted_labels = []

    root = tk.Tk()
    root.title("Detected Faces")

    canvas = Canvas(root, bg="white", scrollregion=(0, 0, 800, 600))
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

    scrollbar = Scrollbar(root, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=Y)
    canvas.config(yscrollcommand=scrollbar.set)

    frame = tk.Frame(canvas, bg="white")
    canvas.create_window((0, 0), window=frame, anchor=tk.NW)

    total_height = 0  # Track the total height of all images and labels

    for filename in os.listdir(test_folder):
        test_image_path = os.path.join(test_folder, filename)

        test_image = face_recognition.load_image_file(test_image_path)
        face_locations = face_recognition.face_locations(test_image)

        if len(face_locations) > 0:
            save_faces_from_image(test_image, face_locations, output_folder)

            for i in range(1, len(face_locations) + 1):
                face_image_path = os.path.join(output_folder, f"face_{i}.jpg")
                face_image = face_recognition.load_image_file(face_image_path)
                face_encodings = face_recognition.face_encodings(face_image)

                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    predictions = face_model.predict([face_encoding])
                    predicted_label = predictions[0]
                    true_label = filename.split('_')[0]

                    true_labels.append(true_label)
                    predicted_labels.append(predicted_label)

                    image = ImageTk.PhotoImage(Image.fromarray(face_image))
                    label_text = f"True Label: {true_label}\nPredicted Label: {predicted_label}"
                    label = Label(frame, text=label_text, image=image, compound="top")
                    label.image = image
                    label.grid(row=total_height, column=0, padx=10, pady=10)

                    total_height += 1

    # Adjust the width (800) and height (120) as needed
    canvas.config(scrollregion=(0, 0, 800, total_height * 170))

    root.mainloop()

    print("True labels:", true_labels)
    print("Predicted labels:", predicted_labels)

if __name__ == "__main__":
    test_images_folder = r"D:\\Computer Vision\\test_image"  # Adjust the path accordingly
    trained_model_path = r"D:\\Computer Vision\\face_detection_model.joblib"  # Adjust the path accordingly
    trained_encoder_path = r"D:\\Computer Vision\\face_encoder_labels.joblib"  # Adjust the path accordingly
    output_faces_folder = r"D:\\Computer Vision\\detected_faces"  # Adjust the path accordingly

    test_face_detection_model(test_images_folder, trained_model_path, trained_encoder_path, output_faces_folder)
