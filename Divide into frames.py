import cv2
import dlib
import os

def divide_video_frames(input_video_path, output_folder):
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

    # Create face detector
    detector = dlib.get_frontal_face_detector()

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

        # Detect faces in the frame
        faces = detector(frame)

        # Save each frame in the video's subfolder
        output_path = os.path.join(video_output_folder, f"frame_{frame_count // (fps // frames_per_second)}.jpg")
        cv2.imwrite(output_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

def process_videos(input_folder, output_folder):
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4") or filename.endswith(".mov") or filename.endswith(".MOV"):
            input_video_path = os.path.join(input_folder, filename)
            divide_video_frames(input_video_path, output_folder)

if __name__ == "__main__":
    # Specify input folder containing both MP4 and MOV videos
    input_folder = r"D:\\Computer Vision\\Dataset"  # Use raw string to handle backslashes
    output_folder = r"D:\\Computer Vision\\frames"  # Use raw string to handle backslashes

    process_videos(input_folder, output_folder)
