from flask import Flask, request, jsonify
from PIL import Image
import face_recognition
import joblib
import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit (adjust as needed)
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

        # Prepare detected faces data
        detected_faces = []
        for i, (face_location, prediction) in enumerate(zip(face_locations, predictions)):
            top, right, bottom, left = face_location
            label = face_model.classes_[prediction.argmax()]
            detected_faces.append({"label": label, "top": top, "right": right, "bottom": bottom, "left": left})

        return detected_faces
    else:
        return None

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        # Save the uploaded file with the full path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        full_file_path = os.path.abspath(file_path)
        file.save(full_file_path)

        # Call face detection function with the full_file_path
        detected_faces = detect_faces(full_file_path)

        return jsonify({"success": "File uploaded and faces detected successfully", "detected_faces": detected_faces})

    return jsonify({"error": "Invalid file format"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
