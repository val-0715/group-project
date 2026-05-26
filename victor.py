import argparse
import os
import platform
import queue
import sys
import threading
import time
import zipfile

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_FILE = "pose_landmarker_lite.task"
FACE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_FILE = "face_landmarker.task"

# Visual tuning
HULL_ALPHA           = 0.16
HULL_COLOR           = (0, 255, 0)
POSE_LINE_THICKNESS  = 3
POSE_LANDMARK_RADIUS = 5
POSE_LANDMARK_COLOR  = (0, 200, 255)
FACE_LINE_COLOR      = (255, 128, 0)
FACE_LANDMARK_RADIUS = 1
FACE_LANDMARK_COLOR  = (0, 128, 255)
VISIBILITY_THRESHOLD = 0.5

_BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

# ---------------------------------------------------------------------------
# Overlay UI constants
# ---------------------------------------------------------------------------
PANEL_W      = 260
PANEL_H      = 220
PANEL_PAD    = 16
BTN_H        = 48
BTN_RADIUS   = 10
FONT         = cv2.FONT_HERSHEY_SIMPLEX

# Colour palette  (BGR)
C_BG         = (20,  20,  20)
C_BORDER     = (60,  60,  60)
C_ON         = (50, 200,  80)
C_OFF        = (50,  50,  50)
C_ON_TEXT    = (255, 255, 255)
C_OFF_TEXT   = (160, 160, 160)
C_TITLE      = (200, 200, 200)
C_HINT       = (100, 100, 100)
C_TOGGLE_KEY = (0,  180, 255)


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
def download_model(url: str, file_name: str) -> str | None:
    if os.path.exists(file_name):
        try:
            if zipfile.is_zipfile(file_name):
                return file_name
            print(f"Model file {file_name} is invalid — re-downloading.")
            os.remove(file_name)
        except Exception:
            try: os.remove(file_name)
            except Exception: pass

    try:
        import urllib.request
        print(f"Downloading model: {file_name}...")
        with urllib.request.urlopen(url) as resp, open(file_name, "wb") as out:
            while chunk := resp.read(8192):
                out.write(chunk)
        if not zipfile.is_zipfile(file_name):
            print(f"Downloaded {file_name} is not a valid zip archive.")
            return None
        return file_name
    except Exception as e:
        print(f"Could not download model automatically: {e}")
        print(f"Place '{file_name}' next to victor.py manually.")
        return None


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def list_cameras(max_index: int = 20) -> list[int]:
    available = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, _BACKEND)
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available


def open_camera(index: int) -> tuple[cv2.VideoCapture, int]:
    cap = cv2.VideoCapture(index, _BACKEND)
    if cap.isOpened():
        return cap, index
    print(f"Could not open camera {index}.")
    available = list_cameras(20)
    if not available:
        raise RuntimeError("No available cameras found.")
    fallback = available[0]
    print(f"Switching to camera {fallback}.")
    cap = cv2.VideoCapture(fallback, _BACKEND)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open fallback camera {fallback}.")
    return cap, fallback


# ---------------------------------------------------------------------------
# Rounded-rectangle helper
# ---------------------------------------------------------------------------
def filled_rounded_rect(img, x1, y1, x2, y2, r, color, alpha=1.0):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x2-r, y2-r), (x1+r, y2-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        img[:] = overlay[:]


# ---------------------------------------------------------------------------
# Overlay panel drawing
# ---------------------------------------------------------------------------
def draw_overlay(frame: np.ndarray, state: dict) -> dict:
    h, w = frame.shape[:2]

    hint = "[ M ] Menu"
    (hw, hh), _ = cv2.getTextSize(hint, FONT, 0.55, 1)
    hx, hy = w - hw - 14, h - 10
    cv2.putText(frame, hint, (hx, hy), FONT, 0.55, C_TOGGLE_KEY, 1, cv2.LINE_AA)

    if not state["show_panel"]:
        return {}

    px = w - PANEL_W - PANEL_PAD
    py = PANEL_PAD
    panel_overlay = frame.copy()
    filled_rounded_rect(panel_overlay, px, py, px + PANEL_W, py + PANEL_H, 12, C_BG)
    cv2.addWeighted(panel_overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (px, py), (px + PANEL_W, py + PANEL_H), C_BORDER, 1)

    title = "TRACKING OPTIONS"
    (tw, _), _ = cv2.getTextSize(title, FONT, 0.5, 1)
    cv2.putText(frame, title, (px + (PANEL_W - tw)//2, py + 26),
                FONT, 0.5, C_TITLE, 1, cv2.LINE_AA)
    cv2.line(frame, (px + 12, py + 36), (px + PANEL_W - 12, py + 36), C_BORDER, 1)

    hit_boxes = {}
    buttons = [
        ("track_body", "Body / Pose"),
        ("track_face", "Face Mesh"),
    ]

    for i, (key, label) in enumerate(buttons):
        bx1 = px + 16
        by1 = py + 50 + i * (BTN_H + 12)
        bx2 = px + PANEL_W - 16
        by2 = by1 + BTN_H
        active = state[key]

        btn_overlay = frame.copy()
        filled_rounded_rect(btn_overlay, bx1, by1, bx2, by2, BTN_RADIUS,
                             C_ON if active else C_OFF)
        cv2.addWeighted(btn_overlay, 0.9, frame, 0.1, 0, frame)

        if active:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), C_ON, 2)

        dot_x = bx1 + 20
        dot_y = (by1 + by2) // 2
        cv2.circle(frame, (dot_x, dot_y), 7, C_ON if active else (80,80,80), -1)
        if active:
            cv2.circle(frame, (dot_x, dot_y), 4, (255,255,255), -1)

        text_col = C_ON_TEXT if active else C_OFF_TEXT
        status    = "ON" if active else "OFF"
        cv2.putText(frame, label, (dot_x + 16, dot_y - 5),
                    FONT, 0.52, text_col, 1, cv2.LINE_AA)
        cv2.putText(frame, status, (dot_x + 16, dot_y + 13),
                    FONT, 0.4, C_ON if active else (80,80,80), 1, cv2.LINE_AA)

        hit_boxes[key] = (bx1, by1, bx2, by2)

    cv2.putText(frame, "Click button or press B / F to toggle",
                (px + 10, py + PANEL_H - 10),
                FONT, 0.38, C_HINT, 1, cv2.LINE_AA)

    return hit_boxes


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_info(frame: np.ndarray, camera_index: int, fps: float) -> None:
    lines = [f"Camera: {camera_index}", f"FPS: {fps:.0f}"]
    for i, text in enumerate(lines):
        y = 30 + i * 30
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
        cv2.rectangle(frame, (8, y - th - 4), (8 + tw + 4, y + 4), (0,0,0), -1)
        cv2.putText(frame, text, (10, y), FONT, 0.8, (255,255,255), 2, cv2.LINE_AA)


def draw_pose(annotated, pose_landmarks_list, width, height):
    for pose_landmarks in pose_landmarks_list:
        points  = [(int(lm.x * width), int(lm.y * height)) for lm in pose_landmarks]
        visible = [p for p, lm in zip(points, pose_landmarks)
                   if lm.visibility > VISIBILITY_THRESHOLD]
        if len(visible) >= 3:
            hull    = cv2.convexHull(np.array(visible, dtype=np.int32))
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [hull], HULL_COLOR)
            cv2.addWeighted(overlay, HULL_ALPHA, annotated, 1 - HULL_ALPHA, 0, annotated)
        for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS:
            s, e = conn.start, conn.end
            if s < len(points) and e < len(points):
                cv2.line(annotated, points[s], points[e], HULL_COLOR, POSE_LINE_THICKNESS)
        for x, y in points:
            cv2.circle(annotated, (x, y), POSE_LANDMARK_RADIUS, POSE_LANDMARK_COLOR, -1)


def draw_face(annotated, face_landmarks_list, width, height):
    connection_sets = [
        vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_NOSE,
    ]
    for face_landmarks in face_landmarks_list:
        landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks]
        for cs in connection_sets:
            for conn in cs:
                s, e = conn.start, conn.end
                if s < len(landmarks) and e < len(landmarks):
                    cv2.line(annotated, landmarks[s], landmarks[e], FACE_LINE_COLOR, 1)
        for x, y in landmarks:
            cv2.circle(annotated, (x, y), FACE_LANDMARK_RADIUS, FACE_LANDMARK_COLOR, -1)


# ---------------------------------------------------------------------------
# Inference thread  (pose + face)
# ---------------------------------------------------------------------------
class PoseFaceThread(threading.Thread):
    def __init__(self, pose_model_path, face_model_path,
                 frame_queue: queue.Queue, result_queue: queue.Queue):
        super().__init__(daemon=True)
        self.pose_model_path = pose_model_path
        self.face_model_path = face_model_path
        self.frame_queue     = frame_queue
        self.result_queue    = result_queue
        self._stop           = threading.Event()
        self._ts             = 0

    def stop(self): self._stop.set()

    def run(self):
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        face_landmarker = None
        if self.face_model_path:
            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.face_model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        try:
            while not self._stop.is_set():
                try:
                    frame, wall_ms = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self._ts = max(self._ts + 1, wall_ms)
                ts       = self._ts
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                try:    pose_result = landmarker.detect_for_video(mp_image, ts)
                except Exception as e:
                    print(f"[Pose] {e}"); pose_result = None

                face_result = None
                if face_landmarker:
                    try:    face_result = face_landmarker.detect_for_video(mp_image, ts)
                    except Exception as e: print(f"[Face] {e}")

                if not self.result_queue.full():
                    self.result_queue.put((pose_result, face_result))
        finally:
            for t in (landmarker, face_landmarker):
                try:
                    if t: t.close()
                except Exception: pass


# ---------------------------------------------------------------------------
# Mouse callback state
# ---------------------------------------------------------------------------
_mouse_click: dict = {"x": -1, "y": -1, "fired": False}

def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_click["x"]     = x
        _mouse_click["y"]     = y
        _mouse_click["fired"] = True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_camera(camera_index: int) -> int:
    pose_model_path = download_model(POSE_MODEL_URL, POSE_MODEL_FILE)
    if not pose_model_path:
        return 1
    face_model_path = download_model(FACE_MODEL_URL, FACE_MODEL_FILE)
    if not face_model_path:
        print("Face model unavailable — continuing without face tracking.")

    try:
        cap, camera_index = open_camera(camera_index)
    except RuntimeError as e:
        print(e); return 1

    pf_frame_q:  queue.Queue = queue.Queue(maxsize=2)
    pf_result_q: queue.Queue = queue.Queue(maxsize=2)

    pf_thread = PoseFaceThread(pose_model_path, face_model_path, pf_frame_q, pf_result_q)
    pf_thread.start()

    window_name = "Victor — Body & Face Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse)

    ui = {
        "show_panel": True,
        "track_body": True,
        "track_face": True,
    }

    fullscreen  = False
    prev_time   = time.time()
    latest_pose = None
    latest_face = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read frame."); break

            wall_ms    = int(time.time() * 1000)
            frame_copy = frame.copy()

            if not pf_frame_q.full():
                pf_frame_q.put_nowait((frame_copy, wall_ms))

            try:
                while True:
                    latest_pose, latest_face = pf_result_q.get_nowait()
            except queue.Empty:
                pass

            annotated  = frame.copy()
            h, w       = annotated.shape[:2]

            if ui["track_body"] and latest_pose and latest_pose.pose_landmarks:
                draw_pose(annotated, latest_pose.pose_landmarks, w, h)

            if ui["track_face"] and latest_face and latest_face.face_landmarks:
                draw_face(annotated, latest_face.face_landmarks, w, h)

            now       = time.time()
            fps       = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now
            draw_info(annotated, camera_index, fps)

            hit_boxes = draw_overlay(annotated, ui)
            cv2.imshow(window_name, annotated)

            if _mouse_click["fired"]:
                mx, my = _mouse_click["x"], _mouse_click["y"]
                _mouse_click["fired"] = False
                for btn_key, (bx1, by1, bx2, by2) in hit_boxes.items():
                    if bx1 <= mx <= bx2 and by1 <= my <= by2:
                        ui[btn_key] = not ui[btn_key]

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                ui["show_panel"] = not ui["show_panel"]
            elif key == ord("b"):
                ui["track_body"] = not ui["track_body"]
            elif key == ord("f"):
                ui["track_face"] = not ui["track_face"]
            elif key == ord("F"):
                fullscreen = not fullscreen
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)
                if not fullscreen:
                    cv2.resizeWindow(window_name, 1280, 720)

    finally:
        pf_thread.stop()
        pf_thread.join(timeout=3)
        try: cap.release()
        except Exception: pass
        cv2.destroyAllWindows()

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Victor — Body & Face Tracker")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if args.list_cameras:
        cams = list_cameras(20)
        print("Available cameras:", ", ".join(map(str, cams)) if cams else "none found")
        return 0

    return run_camera(args.camera)


if __name__ == "__main__":
    sys.exit(main())