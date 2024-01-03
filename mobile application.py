from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import face_recognition
import joblib
import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit (adjust as needed)
UPLOAD_FOLDER = 'uploads'
DETECTED_FOLDER = 'detected_faces'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER

# Load the pre-trained face detection model and encoder
trained_model_path = "face_detection_model.joblib"
trained_encoder_path = "face_encoder_labels.joblib"

face_encodings = joblib.load(trained_model_path)
labels = joblib.load(trained_encoder_path)

# Create a machine learning model for face recognition
face_model = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=True))
face_model.fit(face_encodings, labels)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ...

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

            # Create a separate image for each detected face
            face_image = test_image[top:bottom, left:right]

            # Save the individual face image with the label
            individual_image_path = os.path.join(app.config['DETECTED_FOLDER'], f"face_{i+1}_{label}.jpg")
            individual_pil_image = Image.fromarray(face_image)
            individual_draw = ImageDraw.Draw(individual_pil_image)
            individual_draw.text((0, 0), label, fill="red")  # Add label below the face

            individual_pil_image.save(individual_image_path)

            detected_faces.append({
                "label": label,
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left,
                "individual_image_path": individual_image_path
            })

        # Save the image with squares to the detected folder
        pil_image = Image.fromarray(test_image)
        draw = ImageDraw.Draw(pil_image)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Save the image with squares to the detected folder
        detected_image_path = os.path.join(app.config['DETECTED_FOLDER'], os.path.basename(image_path))
        pil_image.save(detected_image_path)

        # Convert the image to base64
        with open(detected_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        return detected_faces, detected_image_path, encoded_image
    else:
        return None, None, None

# ...

    # Load the test image
    test_image = face_recognition.load_image_file(image_path)

    # Find face locations in the image
    face_locations = face_recognition.face_locations(test_image)

    if len(face_locations) > 0:
        # Draw squares around the detected faces
        pil_image = Image.fromarray(test_image)
        draw = ImageDraw.Draw(pil_image)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Save the image with squares to the detected folder
        detected_image_path = os.path.join(app.config['DETECTED_FOLDER'], os.path.basename(image_path))
        pil_image.save(detected_image_path)

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

        # Convert the image to base64
        with open(detected_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        return detected_faces, detected_image_path, encoded_image
    else:
        return None, None, None

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
        detected_faces, detected_image_path, encoded_image = detect_faces(full_file_path)

        if detected_faces is not None:
            return jsonify({
                "success": "File uploaded and faces detected successfully",
                "detected_faces": detected_faces,
                "detected_image_path": detected_image_path,
                "encoded_image": encoded_image
            })
        else:
            return jsonify({"error": "No faces detected in the uploaded image"})

    return jsonify({"error": "Invalid file format"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
