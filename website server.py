from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import face_recognition
import joblib
import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from flask import jsonify

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DETECTED_FOLDER = 'detected_faces'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER

# Load the pre-trained face detection model and encoder
trained_model_path = "D:\\Computer Vision\\face_detection_model.joblib"
trained_encoder_path = "D:\\Computer Vision\\face_encoder_labels.joblib"

face_encodings = joblib.load(trained_model_path)
labels = joblib.load(trained_encoder_path)

# Create a machine learning model for face recognition
face_model = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=True))
face_model.fit(face_encodings, labels)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_faces_from_image(image, face_locations, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_image = Image.fromarray(image[top:bottom, left:right])
        output_path = os.path.join(output_folder, f"face_{i + 1}.jpg")
        face_image.save(output_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        # Save the uploaded file with the full path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        full_file_path = os.path.abspath(file_path)  # Get the absolute path
        file.save(full_file_path)

        # Call your face detection function with the full_file_path
        detected_faces = detect_faces(full_file_path)

        return render_template('index.html', success='File uploaded and faces detected successfully', detected_faces=detected_faces)

    return render_template('index.html', error='Invalid file format')



def detect_faces(image_path):
    # Load the test image
    test_image = face_recognition.load_image_file(image_path)

    # Find face locations in the image
    face_locations = face_recognition.face_locations(test_image)

    if len(face_locations) > 0:
        # Get face encodings
        face_encodings = face_recognition.face_encodings(test_image, face_locations)

        # Predict labels for each face using the machine learning model
        predictions = face_model.predict_proba(face_encodings)

        # Iterate through the detected faces and their predictions
        for i, (face_location, prediction) in enumerate(zip(face_locations, predictions)):
            top, right, bottom, left = face_location
            output_folder = app.config['DETECTED_FOLDER']

            # Save the entire image to the detected folder
            output_path = os.path.join(output_folder, f"face_{i + 1}.jpg")
            Image.fromarray(test_image).save(output_path)

            # Save the face region to the detected folder
            face_image = Image.fromarray(test_image[top:bottom, left:right])
            face_image_path = os.path.join(output_folder, f"face_{i + 1}_region.jpg")
            face_image.save(face_image_path)

            # Use the prediction to label the detected face
            label = face_model.classes_[prediction.argmax()]
            print(f"Detected face {i + 1} with label: {label}")
# Rest of the code remains unchanged


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
