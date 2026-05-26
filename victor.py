import argparse
import glob
import os
import sys
import time

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

CAMERA_BACKEND = cv2.CAP_ANY
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_FILE = "pose_landmarker_lite.task"

# Prefer backends in this order; some constants may not exist on all builds
PREFERRED_BACKENDS = [
    getattr(cv2, 'CAP_V4L2', cv2.CAP_ANY),
    getattr(cv2, 'CAP_FFMPEG', cv2.CAP_ANY),
    getattr(cv2, 'CAP_GSTREAMER', cv2.CAP_ANY),
    cv2.CAP_ANY,
]

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
        cap = None
        for backend in PREFERRED_BACKENDS:
            cap = cv2.VideoCapture(index, backend)
            if cap is None:
                continue
            if not cap.isOpened():
                cap.release()
                continue
            # try to read one frame to confirm the camera works
            ret, _ = cap.read()
            cap.release()
            if ret:
                available.append(index)
                break
    return available


def draw_info(frame, camera_index, fps):
    cv2.putText(frame, f"Camera: {camera_index}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def run_camera(camera_index=None, video_path=None, headless=False, output_path="output.png"):
    model_path = download_model()
    if not model_path:
        return 1

    base_options = BaseOptions(model_asset_path=model_path)

    # Choose running mode depending on whether we're doing video or a single image
    running_mode = (
        VisionTaskRunningMode.VIDEO if not headless else VisionTaskRunningMode.IMAGE
    )

    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = PoseLandmarker.create_from_options(options)

    # Prefer a provided video path; otherwise try camera index with multiple backends
    cap = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
    elif camera_index is not None:
        for backend in PREFERRED_BACKENDS:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap is None:
                continue
            # allow a short warm-up and a couple of read attempts
            if not cap.isOpened():
                cap.release()
                cap = None
                continue
            ok, _ = cap.read()
            if ok:
                break
            else:
                cap.release()
                cap = None
                continue

    if cap is None or not getattr(cap, 'isOpened', lambda: False)():
        if headless:
            # In headless mode, we'll synthesize a single image below
            print("No camera/video available — running headless single-frame test.")
            cap = None
        else:
            print(f"Could not open camera/video (camera={camera_index} video={video_path})")
            if os.name == "posix":
                video_devices = glob.glob("/dev/video*")
                if not video_devices:
                    print("No /dev/video* devices are visible in this environment.")
                    print(
                        "If you are running inside a container, make sure the host webcam is mounted into the container. Eg: docker run --device=/dev/video0:/dev/video0 ..."
                    )
            landmarker.close()
            return 1

    frame_index = 0
    prev_time = time.time()

    while True:
        if cap is None:
            # synthesize a test image for headless mode
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ok = True
        else:
            ok, frame = cap.read()

        if not ok:
            print("Failed to read frame from camera/video. Ending.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(frame_index * 1000 / max(1, (cap.get(cv2.CAP_PROP_FPS) if cap else 30) or 30))
        result = None
        if running_mode == VisionTaskRunningMode.VIDEO:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = landmarker.detect(mp_image)
        frame_index += 1

        annotated = frame.copy()

        if getattr(result, 'pose_landmarks', None):
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

        draw_info(annotated, camera_index if camera_index is not None else video_path, fps)

        if headless:
            # Save a single annotated frame and exit
            cv2.imwrite(output_path, annotated)
            print(f"Wrote headless output to {output_path}")
            break

        cv2.imshow("Webcam Body Tracking Demo", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    if cap:
        cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Webcam body tracking demo")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open")
    parser.add_argument("--video", type=str, default=None, help="Path to video file to use instead of camera")
    parser.add_argument("--backend", type=str, default=None, help="Force OpenCV backend (e.g. V4L2, FFMPEG, GSTREAMER)")
    parser.add_argument("--debug", action="store_true", help="Run camera/backend probe diagnostics and exit")
    parser.add_argument("--headless", action="store_true", help="Run a single-frame headless test and save output")
    parser.add_argument("--output", type=str, default="output.png", help="Headless output path")
    args = parser.parse_args()

    if args.list_cameras:
        cameras = list_cameras(20)
        if cameras:
            print("Available cameras:", ", ".join(map(str, cameras)))
        else:
            print("No cameras found.")
        return 0

    if args.debug:
        # Run a thorough backend probe for the requested camera index
        print("Probing OpenCV backends for camera index", args.camera)
        for name, backend in [
            ("CAP_V4L2", getattr(cv2, 'CAP_V4L2', None)),
            ("CAP_FFMPEG", getattr(cv2, 'CAP_FFMPEG', None)),
            ("CAP_GSTREAMER", getattr(cv2, 'CAP_GSTREAMER', None)),
            ("CAP_ANY", cv2.CAP_ANY),
        ]:
            if backend is None:
                print(f" - {name}: not available in this OpenCV build")
                continue
            print(f" - Trying backend {name} ({backend})")
            try:
                cap = cv2.VideoCapture(args.camera, backend)
            except Exception as e:
                print(f"   Exception opening VideoCapture: {e}")
                continue
            if not cap or not cap.isOpened():
                print("   Not opened")
                try:
                    cap.release()
                except Exception:
                    pass
                continue
            # try to grab/read a frame
            ok, frame = cap.read()
            if not ok or frame is None:
                print("   Opened but failed to read a frame")
            else:
                h, w = frame.shape[:2]
                print(f"   Read frame {w}x{h}")
                sample_out = f"probe_capture_{name}.png"
                try:
                    cv2.imwrite(sample_out, frame)
                    print(f"   Wrote sample frame to {sample_out}")
                except Exception as e:
                    print(f"   Failed to write sample frame: {e}")
            cap.release()
        print("Probe complete.")
        return 0

    return run_camera(camera_index=args.camera, video_path=args.video, headless=args.headless, output_path=args.output)


if __name__ == "__main__":
    sys.exit(main())
