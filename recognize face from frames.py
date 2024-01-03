import cv2
import dlib
import face_recognition
import os

def divide_video_frames(input_video_path, output_folder):
    # Get the video file name without extension
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    # Create a subfolder for each video
    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    # Open video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames to extract per second
    frames_per_second = 2
    frames_to_extract = fps * frames_per_second

    # Process each frame and save it
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if needed
        if frame_count % (fps // frames_per_second) != 0:
            frame_count += 1
            continue

        # Save each frame in the video's subfolder
        output_path = os.path.join(video_output_folder, f"frame_{frame_count // (fps // frames_per_second)}.jpg")
        cv2.imwrite(output_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

def face_recognition_on_frames(input_folder, output_folder):
    # Iterate through all subfolders (each representing a video)
    for video_folder in os.listdir(input_folder):
        video_folder_path = os.path.join(input_folder, video_folder)

        if os.path.isdir(video_folder_path):
            # Create a subfolder for face recognition frames
            face_output_folder = os.path.join(output_folder, video_folder)
            if not os.path.exists(face_output_folder):
                os.makedirs(face_output_folder)

            # Iterate through frames in the video's subfolder
            for filename in os.listdir(video_folder_path):
                frame_path = os.path.join(video_folder_path, filename)

                # Load the frame
                image = face_recognition.load_image_file(frame_path)

                # Find face locations in the frame
                face_locations = face_recognition.face_locations(image)

                # Draw rectangles around the faces and save the frame
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

                # Save the frame with face detection
                output_path = os.path.join(face_output_folder, filename)
                cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # Specify input folder containing video frames
    input_folder = r"D:\\Computer Vision\\frames"  # Use raw string to handle backslashes
    output_folder_faces = r"D:\\Computer Vision\\frames_faces"  # Use raw string to handle backslashes

    face_recognition_on_frames(input_folder, output_folder_faces)
