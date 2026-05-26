import argparse
import math
import os
import sys
import time
 
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

import numpy as np

if mp is not None:
    try:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )
        from mediapipe.tasks.python.vision.pose_landmarker import (
            PoseLandmarker,
            PoseLandmarkerOptions,
        )
        HAS_POSE_TASKS = True
    except Exception:
        HAS_POSE_TASKS = False
else:
    HAS_POSE_TASKS = False

if mp is not None:
    try:
        mp_face_mesh = mp.solutions.face_mesh
        HAS_FACE_MESH = True
    except Exception:
        mp_face_mesh = None
        HAS_FACE_MESH = False
    try:
        mp_drawing = mp.solutions.drawing_utils
    except Exception:
        mp_drawing = None
    try:
        mp_drawing_styles = mp.solutions.drawing_styles
    except Exception:
        mp_drawing_styles = None
else:
    mp_face_mesh = None
    mp_drawing = None
    mp_drawing_styles = None
    HAS_FACE_MESH = False
 
try:
    mp_hands = mp.solutions.hands
    HAS_HANDS = True
except Exception:
    mp_hands = None
    HAS_HANDS = False
 
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_FILE = "pose_landmarker_lite.task"
 
FACE_DETECTOR = None
 
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (13, 11), (14, 12),
]
 
 
def choose_api(backend_name):
    if backend_name is None or backend_name == "auto":
        return cv2.CAP_ANY
    mapping = {
        "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
        "gstreamer": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
        "any": cv2.CAP_ANY,
    }
    return mapping.get(backend_name.lower(), cv2.CAP_ANY)
 
 
def parse_source(source):
    if source is None:
        return 0
    if isinstance(source, int):
        return source
    if str(source).isdigit():
        return int(source)
    return source
 
 
def is_gui_available():
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True
 
 
def download_model():
    if os.path.exists(MODEL_FILE):
        return MODEL_FILE
 
    try:
        import urllib.request
 
        print("Downloading pose model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        return MODEL_FILE
    except Exception as exc:
        print(f"Could not download model automatically: {exc}")
        print(f"Download the model manually and place it next to code.py as {MODEL_FILE}")
        return None
 
 
def open_capture(source, backend_name="auto"):
    source = parse_source(source)
    api_preference = choose_api(backend_name)
    try:
        cap = cv2.VideoCapture(source, api_preference)
    except Exception:
        cap = cv2.VideoCapture(source)
    return cap
 
 
def load_face_detector():
    global FACE_DETECTOR
    if FACE_DETECTOR is not None:
        return FACE_DETECTOR
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return None
    FACE_DETECTOR = detector
    return FACE_DETECTOR
 
 
def detect_faces_opencv(frame):
    detector = load_face_detector()
    if detector is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces
 
 
def draw_pose_annotations(frame, pose_landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
 
 
def draw_face_overlay(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        left_eye = (x + int(w * 0.25), y + int(h * 0.3))
        right_eye = (x + int(w * 0.75), y + int(h * 0.3))
        cv2.circle(image, left_eye, max(8, int(w * 0.07)), (255, 255, 255), -1)
        cv2.circle(image, left_eye, max(4, int(w * 0.03)), (0, 0, 0), -1)
        cv2.circle(image, right_eye, max(8, int(w * 0.07)), (255, 255, 255), -1)
        cv2.circle(image, right_eye, max(4, int(w * 0.03)), (0, 0, 0), -1)
        mouth_center = (x + int(w * 0.5), y + int(h * 0.7))
        mouth_size = max(10, int(w * 0.2))
        cv2.ellipse(image, mouth_center, (mouth_size, int(h * 0.1)), 0, 0, 180, (0, 0, 255), -1)
 
 
def draw_emoji_eyes(image, landmarks, _face_2d_points):
    if landmarks is None:
        return
    img_h, img_w = image.shape[:2]
    left_eye_center_idx = 1
    right_eye_center_idx = 4
    if left_eye_center_idx >= len(landmarks.landmark) or right_eye_center_idx >= len(landmarks.landmark):
        return
    left_eye_x = int(landmarks.landmark[left_eye_center_idx].x * img_w)
    left_eye_y = int(landmarks.landmark[left_eye_center_idx].y * img_h)
    right_eye_x = int(landmarks.landmark[right_eye_center_idx].x * img_w)
    right_eye_y = int(landmarks.landmark[right_eye_center_idx].y * img_h)
    cv2.circle(image, (left_eye_x, left_eye_y), 30, (255, 255, 255), -1)
    cv2.circle(image, (left_eye_x, left_eye_y), 15, (0, 0, 0), -1)
    cv2.circle(image, (right_eye_x, right_eye_y), 30, (255, 255, 255), -1)
    cv2.circle(image, (right_eye_x, right_eye_y), 15, (0, 0, 0), -1)
 
 
def draw_emoji_mouth(image, landmarks, _face_2d_points):
    if landmarks is None:
        return
    img_h, img_w = image.shape[:2]
    indices = [78, 13, 308, 14]
    if max(indices) >= len(landmarks.landmark):
        return
    mouth_points = np.array([
        [int(landmarks.landmark[idx].x * img_w), int(landmarks.landmark[idx].y * img_h)]
        for idx in indices
    ], np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [mouth_points], True, (0, 0, 255), 5)
    cv2.fillPoly(image, [mouth_points], (0, 0, 200))
 
 
# FIX 1: Improved classify_hand_sign — more robust, doesn't depend on thumb x-direction alone
def classify_hand_sign(active_fingers):
    fingers = set(active_fingers)
    if not fingers:
        return "fist"
    if "Thumb" in fingers and len(fingers) == 1:
        return "thumbs_up"
    if "Index" in fingers and "Middle" in fingers and "Ring" not in fingers and "Pinky" not in fingers:
        return "peace"
    if len(fingers) == 5:
        return "happy"
    return "neutral"
 
 
def get_face_bbox_from_landmarks(landmarks, img_w, img_h):
    xs = [lm.x * img_w for lm in landmarks.landmark]
    ys = [lm.y * img_h for lm in landmarks.landmark]
    x0 = int(min(xs))
    y0 = int(min(ys))
    x1 = int(max(xs))
    y1 = int(max(ys))
    return x0, y0, x1 - x0, y1 - y0
 
 
def estimate_emoji_scale(face_bbox, pose_landmarks, img_w, img_h):
    face_w = face_bbox[2]
    if face_w <= 0:
        return 1.0
    shoulder_w = 0
    if pose_landmarks is not None:
        try:
            pose_landmarks_list = getattr(pose_landmarks, "landmark", pose_landmarks)
            if len(pose_landmarks_list) > 12:
                left = pose_landmarks_list[11]
                right = pose_landmarks_list[12]
                shoulder_w = abs(int((left.x - right.x) * img_w))
        except Exception:
            shoulder_w = 0
    combined = face_w
    if shoulder_w:
        combined = (face_w + shoulder_w) / 2.0
    scale = combined / 160.0
    return max(0.6, min(scale, 2.5))
 
 
def draw_emoji_face(image, face_bbox, sign, scale=1.0):
    x, y, w, h = face_bbox
    if w <= 0 or h <= 0:
        return
    cx = x + w // 2
    cy = y + h // 2
    axes = (max(30, int(w * 0.75 * scale)), max(30, int(h * 0.9 * scale)))
    cv2.ellipse(image, (cx, cy), axes, 0, 0, 360, (0, 255, 255), -1)
    eye_y = cy - int(axes[1] * 0.22)
    eye_dx = int(axes[0] * 0.35)
    eye_r = max(8, int(axes[0] * 0.11))
    left_eye = (cx - eye_dx, eye_y)
    right_eye = (cx + eye_dx, eye_y)
    if sign == "surprise":
        cv2.circle(image, left_eye, eye_r, (255, 255, 255), -1)
        cv2.circle(image, right_eye, eye_r, (255, 255, 255), -1)
        cv2.circle(image, left_eye, max(eye_r // 2, 4), (0, 0, 0), -1)
        cv2.circle(image, right_eye, max(eye_r // 2, 4), (0, 0, 0), -1)
        mouth_center = (cx, cy + int(axes[1] * 0.25))
        cv2.circle(image, mouth_center, max(eye_r, 14), (0, 0, 0), -1)
    else:
        if sign == "fist":
            cv2.line(image, (left_eye[0] - eye_r // 2, eye_y), (left_eye[0] + eye_r // 2, eye_y - eye_r // 2), (0, 0, 0), 4)
            cv2.line(image, (right_eye[0] - eye_r // 2, eye_y - eye_r // 2), (right_eye[0] + eye_r // 2, eye_y), (0, 0, 0), 4)
            mouth_y = cy + int(axes[1] * 0.2)
            cv2.line(image, (cx - eye_r, mouth_y), (cx + eye_r, mouth_y), (0, 0, 0), 6)
        elif sign == "thumbs_up":
            cv2.circle(image, left_eye, eye_r, (255, 255, 255), -1)
            cv2.circle(image, right_eye, eye_r // 2, (255, 255, 255), -1)
            cv2.circle(image, left_eye, max(eye_r // 3, 4), (0, 0, 0), -1)
            cv2.circle(image, right_eye, max(eye_r // 4, 4), (0, 0, 0), -1)
            cv2.ellipse(image, (cx, cy + int(axes[1] * 0.25)), (eye_r, eye_r // 2), 0, 10, 170, (0, 0, 0), 6)
        elif sign == "peace":
            cv2.circle(image, left_eye, eye_r, (255, 255, 255), -1)
            cv2.circle(image, right_eye, eye_r, (255, 255, 255), -1)
            cv2.circle(image, left_eye, max(eye_r // 3, 4), (0, 0, 0), -1)
            cv2.circle(image, right_eye, max(eye_r // 3, 4), (0, 0, 0), -1)
            cv2.ellipse(image, (cx, cy + int(axes[1] * 0.25)), (eye_r, eye_r // 2), 0, 10, 170, (0, 0, 0), 6)
        else:
            cv2.circle(image, left_eye, eye_r, (255, 255, 255), -1)
            cv2.circle(image, right_eye, eye_r, (255, 255, 255), -1)
            cv2.circle(image, left_eye, max(eye_r // 3, 4), (0, 0, 0), -1)
            cv2.circle(image, right_eye, max(eye_r // 3, 4), (0, 0, 0), -1)
            cv2.ellipse(image, (cx, cy + int(axes[1] * 0.25)), (eye_r, eye_r // 2), 0, 10, 170, (0, 0, 0), 6)
 
 
def get_head_pose(landmarks, img_w, img_h):
    if landmarks is None:
        return 0, 0, 0
    try:
        nose_tip = (landmarks.landmark[1].x * img_w, landmarks.landmark[1].y * img_h)
        left_eye = (landmarks.landmark[226].x * img_w, landmarks.landmark[226].y * img_h)
        right_eye = (landmarks.landmark[446].x * img_w, landmarks.landmark[446].y * img_h)
        mouth_left = (landmarks.landmark[78].x * img_w, landmarks.landmark[78].y * img_h)
        mouth_right = (landmarks.landmark[308].x * img_w, landmarks.landmark[308].y * img_h)
    except Exception:
        return 0, 0, 0
    avg_eye_x = (left_eye[0] + right_eye[0]) / 2
    yaw = ((nose_tip[0] - avg_eye_x) / img_w) * 60
    pitch = ((nose_tip[1] - ((left_eye[1] + right_eye[1]) / 2)) / img_h) * 40
    roll = ((mouth_left[1] - mouth_right[1]) / (abs(mouth_left[0] - mouth_right[0]) + 1e-6)) * 30
    return pitch, yaw, roll
 
 
def make_dummy_landmarks(norm_x, norm_y, total=500):
    class LandmarkObj:
        pass
    obj = LandmarkObj()
    lm = [LandmarkObj() for _ in range(total)]
    for i in range(total):
        lm[i].x = 0.5
        lm[i].y = 0.5
    idxs = [1, 2, 4, 13, 14, 78, 226, 308, 446]
    for i in idxs:
        if i < total:
            lm[i].x = norm_x
            lm[i].y = norm_y
    obj.landmark = lm
    return obj
 
 
def draw_info(frame, source_label, fps):
    cv2.putText(frame, f"Source: {source_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
 
 
def create_pose_landmarker():
    model_path = download_model()
    if not model_path:
        return None
    base_options = BaseOptions(model_asset_path=model_path)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)
 
 
def run_body_tracking(source, backend, show_gui, headless, output_path, frame_limit=0):
    if not HAS_POSE_TASKS:
        print("Body tracking requires MediaPipe pose tasks, but they are unavailable.")
        return 1
    cap = open_capture(source, backend)
    if not cap.isOpened():
        print(f"Could not open source for body tracking: {source}")
        return 1
    landmarker = create_pose_landmarker()
    if landmarker is None:
        cap.release()
        return 1
    frame_index = 0
    prev_time = time.time()
    last_saved = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(frame_index * 1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 30)))
        frame_index += 1
        annotated = frame.copy()
        if getattr(result, "pose_landmarks", None):
            for pose_landmarks in result.pose_landmarks:
                draw_pose_annotations(annotated, pose_landmarks)
        fps = 1.0 / (time.time() - prev_time) if time.time() > prev_time else 0.0
        prev_time = time.time()
        draw_info(annotated, "Body", fps)
        if headless:
            if not last_saved:
                cv2.imwrite(output_path, annotated)
                print(f"Saved body output to {output_path}")
                last_saved = True
            if frame_limit and frame_index >= frame_limit:
                break
            continue
        if show_gui and is_gui_available():
            cv2.imshow("Body Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            if frame_limit and frame_index >= frame_limit:
                break
    landmarker.close()
    cap.release()
    if is_gui_available():
        cv2.destroyWindow("Body Tracking")
    return 0
 
 
def run_face_tracking(source, backend, show_gui, headless, output_path, frame_limit=0):
    cap = open_capture(source, backend)
    if not cap.isOpened():
        print(f"Could not open source for face tracking: {source}")
        return 1
    use_solution = HAS_FACE_MESH
    if not use_solution:
        print("Face Mesh not available; falling back to OpenCV face detection.")
    frame_index = 0
    prev_time = time.time()
    last_saved = False
    if use_solution:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        face_mesh = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated = frame.copy()
        if use_solution:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )
                    face_emoji = np.zeros_like(frame)
                    draw_emoji_eyes(face_emoji, face_landmarks, None)
                    draw_emoji_mouth(face_emoji, face_landmarks, None)
                    annotated = cv2.addWeighted(frame, 0.6, face_emoji, 0.4, 0)
        else:
            faces = detect_faces_opencv(frame)
            draw_face_overlay(annotated, faces)
        fps = 1.0 / (time.time() - prev_time) if time.time() > prev_time else 0.0
        prev_time = time.time()
        draw_info(annotated, "Face", fps)
        frame_index += 1
        if headless:
            if not last_saved:
                cv2.imwrite(output_path, annotated)
                print(f"Saved face output to {output_path}")
                last_saved = True
            if frame_limit and frame_index >= frame_limit:
                break
            continue
        if show_gui and is_gui_available():
            cv2.imshow("Face Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            if frame_limit and frame_index >= frame_limit:
                break
    if face_mesh is not None:
        face_mesh.close()
    cap.release()
    if is_gui_available():
        cv2.destroyWindow("Face Tracking")
    return 0
 
 
def run_finger_tracking(source, backend, show_gui, headless, output_path, frame_limit=0):
    if not HAS_HANDS:
        print("Finger tracking requires MediaPipe Hands, but it is unavailable.")
        return 1
    cap = open_capture(source, backend)
    if not cap.isOpened():
        print(f"Could not open source for finger tracking: {source}")
        return 1
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    frame_index = 0
    prev_time = time.time()
    last_saved = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (960, 540))
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_small.flags.writeable = False
        results = hands.process(rgb_small)
        rgb_small.flags.writeable = True
        annotated = frame.copy()
        active_fingers = []
        if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # determine handedness if available
                        handedness = None
                        if getattr(results, 'multi_handedness', None) and idx < len(results.multi_handedness):
                            try:
                                handedness = results.multi_handedness[idx].classification[0].label
                            except Exception:
                                handedness = None
                        if mp_drawing is not None:
                            try:
                                mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            except Exception:
                                pass
                        tips = [8, 12, 16, 20]
                        knuckles = [6, 10, 14, 18]
                        finger_names = ["Index", "Middle", "Ring", "Pinky"]
                        for t, k, name in zip(tips, knuckles, finger_names):
                            try:
                                if hand_landmarks.landmark[t].y < hand_landmarks.landmark[k].y:
                                    active_fingers.append(name)
                            except Exception:
                                continue
                        # improved thumb detection using handedness when available
                        try:
                            thumb_tip = hand_landmarks.landmark[4]
                            thumb_ip = hand_landmarks.landmark[3]
                            thumb_mcp = hand_landmarks.landmark[2]
                            if handedness == 'Right':
                                if thumb_tip.x < thumb_ip.x:
                                    active_fingers.append("Thumb")
                            elif handedness == 'Left':
                                if thumb_tip.x > thumb_ip.x:
                                    active_fingers.append("Thumb")
                            else:
                                if abs(thumb_tip.x - thumb_mcp.x) > 0.03:
                                    active_fingers.append("Thumb")
                        except Exception:
                            pass
        cv2.putText(annotated, f"Fingers Up: {len(active_fingers)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if active_fingers:
            fingerprint = ", ".join(active_fingers)
            cv2.putText(annotated, fingerprint, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        fps = 1.0 / (time.time() - prev_time) if time.time() > prev_time else 0.0
        prev_time = time.time()
        draw_info(annotated, "Finger", fps)
        frame_index += 1
        if headless:
            if not last_saved:
                cv2.imwrite(output_path, annotated)
                print(f"Saved finger output to {output_path}")
                last_saved = True
            if frame_limit and frame_index >= frame_limit:
                break
            continue
        if show_gui and is_gui_available():
            cv2.imshow("Finger Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            if frame_limit and frame_index >= frame_limit:
                break
    hands.close()
    cap.release()
    if is_gui_available():
        cv2.destroyWindow("Finger Tracking")
    return 0
 
 
def run_all(source, backend, show_gui, headless, output_path, frame_limit=0):
    if not HAS_POSE_TASKS:
        print("All-mode requires MediaPipe pose tasks. Falling back to face+finger only.")
    cap = open_capture(source, backend)
    if not cap.isOpened():
        print(f"Could not open source for all-mode: {source}")
        return 1
    pose_landmarker = create_pose_landmarker() if HAS_POSE_TASKS else None
    face_mesh = None
    if HAS_FACE_MESH:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    hands = None
    if HAS_HANDS:
        hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    frame_index = 0
    prev_time = time.time()
    last_saved = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated = frame.copy()
        body_pose_landmarks = None
        if pose_landmarker is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = pose_landmarker.detect_for_video(mp_image, int(frame_index * 1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 30)))
            if getattr(result, "pose_landmarks", None):
                body_pose_landmarks = result.pose_landmarks[0]
                for pose_landmarks in result.pose_landmarks:
                    draw_pose_annotations(annotated, pose_landmarks)
 
        # FIX 3: Work on flip consistently; draw hand landmarks onto flip before face emoji
        flip = cv2.flip(frame, 1)
        hand_sign = "neutral"
        active_fingers = []
 
        if hands is not None:
            small_frame = cv2.resize(flip, (960, 540))
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            rgb_small.flags.writeable = False
            hand_results = hands.process(rgb_small)
            rgb_small.flags.writeable = True
            if hand_results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    # determine handedness if available
                    handedness = None
                    if getattr(hand_results, 'multi_handedness', None) and idx < len(hand_results.multi_handedness):
                        try:
                            handedness = hand_results.multi_handedness[idx].classification[0].label
                        except Exception:
                            handedness = None
                    # Draw landmarks onto flipped image
                    if mp_drawing is not None:
                        try:
                            mp_drawing.draw_landmarks(flip, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        except Exception:
                            pass
                    tips = [8, 12, 16, 20]
                    knuckles = [6, 10, 14, 18]
                    finger_names = ["Index", "Middle", "Ring", "Pinky"]
                    for t, k, name in zip(tips, knuckles, finger_names):
                        try:
                            if hand_landmarks.landmark[t].y < hand_landmarks.landmark[k].y:
                                active_fingers.append(name)
                        except Exception:
                            continue
                    # improved thumb detection using handedness when available
                    try:
                        thumb_tip = hand_landmarks.landmark[4]
                        thumb_ip = hand_landmarks.landmark[3]
                        thumb_mcp = hand_landmarks.landmark[2]
                        if handedness == 'Right':
                            if thumb_tip.x < thumb_ip.x:
                                active_fingers.append("Thumb")
                        elif handedness == 'Left':
                            if thumb_tip.x > thumb_ip.x:
                                active_fingers.append("Thumb")
                        else:
                            if abs(thumb_tip.x - thumb_mcp.x) > 0.03:
                                active_fingers.append("Thumb")
                    except Exception:
                        pass
            hand_sign = classify_hand_sign(active_fingers)
 
        if face_mesh is not None:
            rgb_face = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb_face)
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                face_bbox = get_face_bbox_from_landmarks(face_landmarks, flip.shape[1], flip.shape[0])
                emoji_scale = estimate_emoji_scale(face_bbox, body_pose_landmarks, flip.shape[1], flip.shape[0])
                draw_emoji_face(flip, face_bbox, hand_sign, emoji_scale)
            annotated = cv2.flip(flip, 1)
        else:
            faces = detect_faces_opencv(flip)
            for face in faces:
                emoji_scale = estimate_emoji_scale(face, body_pose_landmarks, flip.shape[1], flip.shape[0])
                draw_emoji_face(flip, face, hand_sign, emoji_scale)
            annotated = cv2.flip(flip, 1)
 
        cv2.putText(annotated, f"Fingers Up: {len(active_fingers)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if active_fingers:
            cv2.putText(annotated, ", ".join(active_fingers), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
 
        fps = 1.0 / (time.time() - prev_time) if time.time() > prev_time else 0.0
        prev_time = time.time()
        draw_info(annotated, "All", fps)
        frame_index += 1
        if headless:
            if not last_saved:
                cv2.imwrite(output_path, annotated)
                print(f"Saved combined output to {output_path}")
                last_saved = True
            if frame_limit and frame_index >= frame_limit:
                break
            continue
        if show_gui and is_gui_available():
            cv2.imshow("Combined Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            if frame_limit and frame_index >= frame_limit:
                break
    if pose_landmarker is not None:
        pose_landmarker.close()
    if face_mesh is not None:
        face_mesh.close()
    if hands is not None:
        hands.close()
    cap.release()
    if is_gui_available():
        cv2.destroyWindow("Combined Tracking")
    return 0
 
 
def list_cameras(max_index=10):
    available = []
    for index in range(max_index + 1):
        cap = open_capture(index)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            continue
        ret, _ = cap.read()
        if ret:
            available.append(index)
        cap.release()
    return available
 
 
def main():
    parser = argparse.ArgumentParser(description="Unified body, face, and finger tracking demo")
    parser.add_argument("--mode", choices=["body", "face", "finger", "all"], default="all", help="Which tracking mode to run")
    parser.add_argument("--camera", default=0, help="Camera index or device path to use")
    parser.add_argument("--input", default=None, help="Video file input path (overrides --camera)")
    parser.add_argument("--backend", default="auto", help="OpenCV backend to use: auto, v4l2, dshow, avfoundation, gstreamer")
    parser.add_argument("--show-gui", action="store_true", default=True, help="Show OpenCV display windows when available")
    parser.add_argument("--headless", action="store_true", help="Run a short headless processing test and optionally save one annotated frame")
    parser.add_argument("--output", default="output.png", help="Output path when running headless")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera indices and exit")
    args = parser.parse_args()
 
    if args.list_cameras:
        cams = list_cameras(20)
        if cams:
            print("Available cameras:", ", ".join(map(str, cams)))
        else:
            print("No cameras found.")
        return 0
 
    source = args.input if args.input else args.camera
    frame_limit = 30 if args.headless else 0
    if args.mode == "body":
        return run_body_tracking(source, args.backend, args.show_gui, args.headless, args.output, frame_limit)
    if args.mode == "face":
        return run_face_tracking(source, args.backend, args.show_gui, args.headless, args.output, frame_limit)
    if args.mode == "finger":
        return run_finger_tracking(source, args.backend, args.show_gui, args.headless, args.output, frame_limit)
    return run_all(source, args.backend, args.show_gui, args.headless, args.output, frame_limit)
 
 
if __name__ == "__main__":
    sys.exit(main())