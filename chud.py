# python -m pip install --upgrade pip
# python -m pip install opencv-python mediapipe
# python -m pip install --upgrade pip setuptools wheel
# python -m pip install opencv-python mediapipe

import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_FILE = "pose_landmarker_lite.task"


def download_model():
    if os.path.exists(MODEL_FILE):
        return MODEL_FILE

    try:
        import urllib.request
        print("Downloading pose model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        return MODEL_FILE
    except Exception as e:
        print(f"Could not download model automatically: {e}")
        print("Download the model manually and place it next to main.py as pose_landmarker_lite.task")
        return None


def list_cameras(max_index=20):
    available = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(index)
        cap.release()
    return available


def draw_info(frame, camera_index, fps):
    cv2.putText(frame, f"Camera: {camera_index}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def run_camera(camera_index):
    model_path = download_model()
    if not model_path:
        return 1

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}")
        return 1

    frame_index = 0
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(frame_index * 1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 30))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_index += 1

        annotated = frame.copy()

        if result.pose_landmarks:
            for pose_landmarks in result.pose_landmarks:
                points = [(int(lm.x * annotated.shape[1]), int(lm.y * annotated.shape[0])) for lm in pose_landmarks]

                connections = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                    (24, 26), (26, 28), (25, 23), (26, 24), (27, 29), (29, 31),
                    (28, 30), (30, 32), (13, 11), (14, 12)
                ]

                for a, b in connections:
                    if a < len(points) and b < len(points):
                        cv2.line(annotated, points[a], points[b], (0, 255, 0), 2)

                for x, y in points:
                    cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time > prev_time else 0.0
        prev_time = current_time

        draw_info(annotated, camera_index, fps)
        cv2.imshow("Webcam Body Tracking Demo", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Webcam body tracking demo")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open")
    args = parser.parse_args()

    if args.list_cameras:
        cameras = list_cameras(20)
        if cameras:
            print("Available cameras:", ", ".join(map(str, cameras)))
        else:
            print("No cameras found.")
        return 0

    return run_camera(args.camera)


if __name__ == "__main__":
    sys.exit(main())