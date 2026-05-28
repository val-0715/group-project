# python version needed is Python 3.11.x  (3.11.9 recommended) you will be able to download it on the python page make sure when downloading to add python.exe to path
# py -3.11 -m pip install mediapipe==0.10.14 opencv-python numpy

import argparse
import os
import sys
import time
 
import cv2
import mediapipe as mp
import numpy as np
 
# ── Pose Landmarker (Tasks API) ──────────────────────────────────────────────
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
 
# ── Face Mesh ────────────────────────────────────────────────────────────────
try:
    mp_face_mesh       = mp.solutions.face_mesh
    mp_drawing         = mp.solutions.drawing_utils
    mp_drawing_styles  = mp.solutions.drawing_styles
    HAS_FACE_MESH      = True
except Exception:
    mp_face_mesh = mp_drawing = mp_drawing_styles = None
    HAS_FACE_MESH = False
 
# ── Hands ────────────────────────────────────────────────────────────────────
try:
    mp_hands  = mp.solutions.hands
    HAS_HANDS = True
except Exception:
    mp_hands  = None
    HAS_HANDS = False
 
# ── Constants ────────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_FILE = "pose_landmarker_lite.task"
 
FACE_DETECTOR = None
 
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),
    (24,26),(26,28),(27,29),(28,30),(29,31),(30,32),
    (13,11),(14,12),
]
 
SIGN_LABELS = {
    "fist":      "Fist      -> Angry",
    "thumbs_up": "Thumb Up  -> Wink",
    "peace":     "Peace     -> Happy",
    "surprise":  "All 5     -> Surprise",
    "neutral":   "Other     -> Neutral",
}
 
# ── Helpers ──────────────────────────────────────────────────────────────────
def choose_api(backend_name):
    if backend_name is None or backend_name == "auto":
        return cv2.CAP_ANY
    mapping = {
        "v4l2":        getattr(cv2, "CAP_V4L2",        cv2.CAP_ANY),
        "dshow":       getattr(cv2, "CAP_DSHOW",       cv2.CAP_ANY),
        "avfoundation":getattr(cv2, "CAP_AVFOUNDATION",cv2.CAP_ANY),
        "gstreamer":   getattr(cv2, "CAP_GSTREAMER",   cv2.CAP_ANY),
        "any":         cv2.CAP_ANY,
    }
    return mapping.get(backend_name.lower(), cv2.CAP_ANY)
 
 
def parse_source(source):
    if source is None:        return 0
    if isinstance(source,int):return source
    if str(source).isdigit(): return int(source)
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
        print("Downloading pose model…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        return MODEL_FILE
    except Exception as exc:
        print(f"Could not download model: {exc}")
        return None
 
 
def open_capture(source, backend_name="auto"):
    source = parse_source(source)
    try:
        cap = cv2.VideoCapture(source, choose_api(backend_name))
    except Exception:
        cap = cv2.VideoCapture(source)
    return cap
 
 
def load_face_detector():
    global FACE_DETECTOR
    if FACE_DETECTOR is not None:
        return FACE_DETECTOR
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det  = cv2.CascadeClassifier(path)
    if det.empty():
        return None
    FACE_DETECTOR = det
    return FACE_DETECTOR
 
 
def detect_faces_opencv(frame):
    det = load_face_detector()
    if det is None:
        return []
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    return faces
 
 
def make_resizable_window(name, w=1280, h=720):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
 
 
def toggle_fullscreen(name):
    cur = cv2.getWindowProperty(name, cv2.WND_PROP_FULLSCREEN)
    if cur == cv2.WINDOW_FULLSCREEN:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    else:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
 
 
# ── Drawing ──────────────────────────────────────────────────────────────────
def draw_pose_annotations(frame, pose_landmarks):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x*w), int(lm.y*h)) for lm in pose_landmarks]
    for a,b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0,255,0), 2)
    for x,y in pts:
        cv2.circle(frame, (x,y), 3, (0,255,255), -1)
 
 
def draw_info(frame, label, fps):
    cv2.putText(frame, f"Source: {label}", (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}",  (10,60),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
 
 
def draw_sign_label(frame, sign):
    label = SIGN_LABELS.get(sign, sign)
    fw    = frame.shape[1]
    (tw,_),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, label, (fw-tw-10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
 
 
def get_face_bbox_from_landmarks(landmarks, img_w, img_h):
    xs = [lm.x*img_w for lm in landmarks.landmark]
    ys = [lm.y*img_h for lm in landmarks.landmark]
    x0,y0 = int(min(xs)), int(min(ys))
    x1,y1 = int(max(xs)), int(max(ys))
    return x0, y0, x1-x0, y1-y0
 
 
def estimate_emoji_scale(face_bbox, pose_landmarks, img_w, img_h):
    face_w = face_bbox[2]
    if face_w <= 0:
        return 1.0
    shoulder_w = 0
    if pose_landmarks is not None:
        try:
            pll = getattr(pose_landmarks, "landmark", pose_landmarks)
            if len(pll) > 12:
                shoulder_w = abs(int((pll[11].x - pll[12].x) * img_w))
        except Exception:
            pass
    combined = face_w if not shoulder_w else (face_w + shoulder_w) / 2.0
    return max(0.6, min(combined / 160.0, 2.5))
 
 
def draw_emoji_face(image, face_bbox, sign, scale=1.0):
    x, y, w, h = face_bbox
    if w <= 0 or h <= 0:
        return
    cx   = x + w // 2
    cy   = y + h // 2
    axes = (max(30, int(w * 0.75 * scale)), max(30, int(h * 0.9 * scale)))
 
    colours = {
        "fist":      (60,  60,  220),
        "thumbs_up": (0,   180, 255),
        "peace":     (0,   200, 120),
        "surprise":  (0,   255, 255),
        "neutral":   (180, 200, 0  ),
    }
    cv2.ellipse(image, (cx,cy), axes, 0, 0, 360, colours.get(sign,(0,255,255)), -1)
 
    eye_y     = cy - int(axes[1] * 0.22)
    eye_dx    = int(axes[0] * 0.35)
    eye_r     = max(8, int(axes[0] * 0.11))
    left_eye  = (cx - eye_dx, eye_y)
    right_eye = (cx + eye_dx, eye_y)
    mouth_cy  = cy + int(axes[1] * 0.25)
 
    if sign == "fist":
        # X eyes + flat mouth = angry
        for ex,ey in [left_eye, right_eye]:
            cv2.line(image,(ex-eye_r,ey-eye_r),(ex+eye_r,ey+eye_r),(0,0,0),4)
            cv2.line(image,(ex+eye_r,ey-eye_r),(ex-eye_r,ey+eye_r),(0,0,0),4)
        cv2.line(image,(cx-eye_r*2,mouth_cy),(cx+eye_r*2,mouth_cy),(0,0,0),6)
 
    elif sign == "thumbs_up":
        # One eye open, one wink + smile
        cv2.circle(image, left_eye,  eye_r, (255,255,255), -1)
        cv2.circle(image, left_eye,  max(eye_r//3,4), (0,0,0), -1)
        cv2.line(image,
                 (right_eye[0]-eye_r, right_eye[1]),
                 (right_eye[0]+eye_r, right_eye[1]), (0,0,0), 4)
        cv2.ellipse(image,(cx,mouth_cy),(eye_r*2,eye_r),0,10,170,(0,0,0),6)
 
    elif sign == "peace":
        # Open eyes + big smile
        cv2.circle(image, left_eye,  eye_r, (255,255,255), -1)
        cv2.circle(image, right_eye, eye_r, (255,255,255), -1)
        cv2.circle(image, left_eye,  max(eye_r//3,4), (0,0,0), -1)
        cv2.circle(image, right_eye, max(eye_r//3,4), (0,0,0), -1)
        cv2.ellipse(image,(cx,mouth_cy),(eye_r*2,eye_r),0,10,170,(0,0,0),6)
 
    elif sign == "surprise":
        # Big round eyes + O mouth
        cv2.circle(image, left_eye,  eye_r+4, (255,255,255), -1)
        cv2.circle(image, right_eye, eye_r+4, (255,255,255), -1)
        cv2.circle(image, left_eye,  max(eye_r//2,5), (0,0,0), -1)
        cv2.circle(image, right_eye, max(eye_r//2,5), (0,0,0), -1)
        cv2.circle(image, (cx,mouth_cy), max(eye_r+4,16), (0,0,0), -1)
 
    else:  # neutral
        cv2.circle(image, left_eye,  eye_r, (255,255,255), -1)
        cv2.circle(image, right_eye, eye_r, (255,255,255), -1)
        cv2.circle(image, left_eye,  max(eye_r//3,4), (0,0,0), -1)
        cv2.circle(image, right_eye, max(eye_r//3,4), (0,0,0), -1)
        cv2.ellipse(image,(cx,mouth_cy),(eye_r*2,eye_r),0,10,170,(0,0,0),6)
 
 
# ── Hand detection helpers ───────────────────────────────────────────────────
def detect_fingers(hand_landmarks):
    """Return list of finger names that are extended."""
    lm             = hand_landmarks.landmark
    active         = []
    finger_tips    = [8,  12, 16, 20]
    finger_pips    = [6,  10, 14, 18]
    finger_names   = ["Index","Middle","Ring","Pinky"]
 
    for tip, pip, name in zip(finger_tips, finger_pips, finger_names):
        if lm[tip].y < lm[pip].y:          # tip above pip = extended
            active.append(name)
 
    # Thumb: compare distance from wrist
    dist_tip = abs(lm[4].x - lm[0].x) + abs(lm[4].y - lm[0].y)
    dist_mcp = abs(lm[2].x - lm[0].x) + abs(lm[2].y - lm[0].y)
    if dist_tip > dist_mcp * 1.2:
        active.append("Thumb")
 
    return active
 
 
def classify_hand_sign(active_fingers):
    fingers = set(active_fingers)
    if not fingers:
        return "fist"
    if fingers == {"Thumb"}:
        return "thumbs_up"
    if "Index" in fingers and "Middle" in fingers and "Ring" not in fingers and "Pinky" not in fingers:
        return "peace"
    if len(fingers) >= 4:
        return "surprise"
    return "neutral"
 
 
def create_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
 
 
# ── Pose landmarker ──────────────────────────────────────────────────────────
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
 
 
# ── Tracking modes ───────────────────────────────────────────────────────────
def run_body_tracking(source, backend, show_gui, headless, output_path, frame_limit=0):
    if not HAS_POSE_TASKS:
        print("Body tracking requires MediaPipe pose tasks.")
        return 1
    cap = open_capture(source, backend)
    if not cap.isOpened():
        print(f"Could not open: {source}")
        return 1
    landmarker = create_pose_landmarker()
    if landmarker is None:
        cap.release(); return 1
 
    WIN = "Body Tracking"
    if show_gui and is_gui_available() and not headless:
        make_resizable_window(WIN)
 
    frame_index = 0
    prev_time   = time.time()
    last_saved  = False
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect_for_video(
            mp_image, int(frame_index*1000/max(1, cap.get(cv2.CAP_PROP_FPS) or 30)))
        frame_index += 1
        annotated = frame.copy()
        if getattr(result,"pose_landmarks",None):
            for pl in result.pose_landmarks:
                draw_pose_annotations(annotated, pl)
        fps = 1.0/(time.time()-prev_time) if time.time()>prev_time else 0.0
        prev_time = time.time()
        draw_info(annotated,"Body",fps)
        if headless:
            if not last_saved:
                cv2.imwrite(output_path, annotated)
                print(f"Saved to {output_path}")
                last_saved = True
            if frame_limit and frame_index>=frame_limit: break
            continue
        if show_gui and is_gui_available():
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("f"): toggle_fullscreen(WIN)
        else:
            if frame_limit and frame_index>=frame_limit: break
 
    landmarker.close(); cap.release(); cv2.destroyAllWindows()
    return 0
 
 
def run_face_tracking(source, backend, show_gui, headless, output_path, frame_limit=0):
    cap = open_capture(source, backend)
    if not cap.isOpened():
        print(f"Could not open: {source}"); return 1
 
    WIN = "Face Tracking"
    if show_gui and is_gui_available() and not headless:
        make_resizable_window(WIN)
 
    face_mesh = None
    if HAS_FACE_MESH:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    else:
        print("Face Mesh unavailable; using OpenCV fallback.")
 
    frame_index = 0
    prev_time   = time.time()
    last_saved  = False
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame     = cv2.flip(frame, 1)
        annotated = frame.copy()
 
        if face_mesh:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                for fl in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, fl, mp_face_mesh.FACEMESH_TESSELATION,
                        None, mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        frame, fl, mp_face_mesh.FACEMESH_CONTOURS,
                        None, mp_drawing_styles.get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        frame, fl, mp_face_mesh.FACEMESH_IRISES,
                        None, mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                annotated = frame
        else:
            draw_face_overlay(annotated, detect_faces_opencv(frame))
 
        fps = 1.0/(time.time()-prev_time) if time.time()>prev_time else 0.0
        prev_time = time.time()
        draw_info(annotated,"Face",fps)
        frame_index += 1
 
        if headless:
            if not last_saved:
                cv2.imwrite(output_path, annotated)
                print(f"Saved to {output_path}")
                last_saved = True
            if frame_limit and frame_index>=frame_limit: break
            continue
        if show_gui and is_gui_available():
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("f"): toggle_fullscreen(WIN)
        else:
            if frame_limit and frame_index>=frame_limit: break
 
    if face_mesh: face_mesh.close()
    cap.release(); cv2.destroyAllWindows()
    return 0
 
 
def run_finger_tracking(source, backend, show_gui, headless, output_path, frame_limit=0):
    if not HAS_HANDS:
        print("Finger tracking requires MediaPipe Hands."); return 1
    cap = open_capture(source, backend)
    if not cap.isOpened():
        print(f"Could not open: {source}"); return 1
 
    WIN   = "Finger Tracking"
    hands = create_hands()
    if show_gui and is_gui_available() and not headless:
        make_resizable_window(WIN)
 
    frame_index = 0
    prev_time   = time.time()
    last_saved  = False
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True
 
        annotated      = frame.copy()
        active_fingers = []
 
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated, hl, mp_hands.HAND_CONNECTIONS)
                active_fingers.extend(detect_fingers(hl))
 
        cv2.putText(annotated, f"Fingers Up: {len(active_fingers)}", (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        if active_fingers:
            cv2.putText(annotated, ", ".join(active_fingers), (30,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
 
        fps = 1.0/(time.time()-prev_time) if time.time()>prev_time else 0.0
        prev_time = time.time()
        draw_info(annotated,"Finger",fps)
        frame_index += 1
 
        if headless:
            if not last_saved:
                cv2.imwrite(output_path, annotated)
                print(f"Saved to {output_path}")
                last_saved = True
            if frame_limit and frame_index>=frame_limit: break
            continue
        if show_gui and is_gui_available():
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("f"): toggle_fullscreen(WIN)
        else:
            if frame_limit and frame_index>=frame_limit: break
 
    hands.close(); cap.release(); cv2.destroyAllWindows()
    return 0
 
 
def run_all(source, backend, show_gui, headless, output_path, frame_limit=0):
    if not HAS_POSE_TASKS:
        print("Pose tasks unavailable — running face+hand only.")
 
    cap = open_capture(source, backend)
    if not cap.isOpened():
        print(f"Could not open: {source}"); return 1
 
    pose_landmarker = create_pose_landmarker() if HAS_POSE_TASKS else None
 
    face_mesh = None
    if HAS_FACE_MESH:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
 
    hands = create_hands() if HAS_HANDS else None
 
    WIN = "Combined Tracking"
    if show_gui and is_gui_available() and not headless:
        make_resizable_window(WIN)
 
    frame_index = 0
    prev_time   = time.time()
    last_saved  = False
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
 
        # ── Pose on original frame ──
        annotated           = frame.copy()
        body_pose_landmarks = None
        if pose_landmarker is not None:
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = pose_landmarker.detect_for_video(
                mp_image,
                int(frame_index*1000/max(1, cap.get(cv2.CAP_PROP_FPS) or 30)))
            if getattr(result,"pose_landmarks",None):
                body_pose_landmarks = result.pose_landmarks[0]
                for pl in result.pose_landmarks:
                    draw_pose_annotations(annotated, pl)
 
        # ── Flip for hand + face (mirror = natural) ──
        flip           = cv2.flip(frame, 1)
        hand_sign      = "fist"
        active_fingers = []
 
        # ── Hand detection ──
        if hands is not None:
            rgb_h = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
            rgb_h.flags.writeable = False
            hand_results = hands.process(rgb_h)
            rgb_h.flags.writeable = True
 
            if hand_results.multi_hand_landmarks:
                for hl in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(flip, hl, mp_hands.HAND_CONNECTIONS)
                    active_fingers.extend(detect_fingers(hl))
 
            hand_sign = classify_hand_sign(active_fingers)
 
        # ── Face emoji driven by hand_sign ──
        if face_mesh is not None:
            rgb_f    = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
            f_result = face_mesh.process(rgb_f)
            if f_result.multi_face_landmarks:
                fl         = f_result.multi_face_landmarks[0]
                face_bbox  = get_face_bbox_from_landmarks(fl, flip.shape[1], flip.shape[0])
                emoji_scale= estimate_emoji_scale(face_bbox, body_pose_landmarks,
                                                  flip.shape[1], flip.shape[0])
                draw_emoji_face(flip, face_bbox, hand_sign, emoji_scale)
            annotated = cv2.flip(flip, 1)
        else:
            faces = detect_faces_opencv(flip)
            for face in faces:
                emoji_scale = estimate_emoji_scale(face, body_pose_landmarks,
                                                   flip.shape[1], flip.shape[0])
                draw_emoji_face(flip, face, hand_sign, emoji_scale)
            annotated = cv2.flip(flip, 1)
 
        # ── HUD ──
        cv2.putText(annotated, f"Fingers Up: {len(active_fingers)}", (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        if active_fingers:
            cv2.putText(annotated, ", ".join(dict.fromkeys(active_fingers)), (30,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
 
        draw_sign_label(annotated, hand_sign)
        fps = 1.0/(time.time()-prev_time) if time.time()>prev_time else 0.0
        prev_time = time.time()
        draw_info(annotated,"All",fps)
        frame_index += 1
 
        if headless:
            if not last_saved:
                cv2.imwrite(output_path, annotated)
                print(f"Saved to {output_path}")
                last_saved = True
            if frame_limit and frame_index>=frame_limit: break
            continue
 
        if show_gui and is_gui_available():
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("f"): toggle_fullscreen(WIN)
        else:
            if frame_limit and frame_index>=frame_limit: break
 
    if pose_landmarker: pose_landmarker.close()
    if face_mesh:       face_mesh.close()
    if hands:           hands.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0
 
 
# ── Camera listing ───────────────────────────────────────────────────────────
def list_cameras(max_index=10):
    available = []
    for i in range(max_index+1):
        cap = open_capture(i)
        if cap and cap.isOpened():
            ret,_ = cap.read()
            if ret: available.append(i)
            cap.release()
    return available
 
 
# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Body / face / finger tracking demo")
    parser.add_argument("--mode",    choices=["body","face","finger","all"], default="all")
    parser.add_argument("--camera",  default=0)
    parser.add_argument("--input",   default=None)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--show-gui",     action="store_true", default=True)
    parser.add_argument("--headless",     action="store_true")
    parser.add_argument("--output",       default="output.png")
    parser.add_argument("--list-cameras", action="store_true")
    args = parser.parse_args()
 
    if args.list_cameras:
        cams = list_cameras(20)
        print("Cameras:", ", ".join(map(str,cams)) if cams else "None found.")
        return 0
 
    source      = args.input if args.input else args.camera
    frame_limit = 30 if args.headless else 0
 
    if args.mode == "body":   return run_body_tracking  (source,args.backend,args.show_gui,args.headless,args.output,frame_limit)
    if args.mode == "face":   return run_face_tracking  (source,args.backend,args.show_gui,args.headless,args.output,frame_limit)
    if args.mode == "finger": return run_finger_tracking(source,args.backend,args.show_gui,args.headless,args.output,frame_limit)
    return run_all(source,args.backend,args.show_gui,args.headless,args.output,frame_limit)
 
 
if __name__ == "__main__":
    sys.exit(main()) 
