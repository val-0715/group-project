# python version needed is Python 3.11.x  (3.11.9 recommended) you will be able to download it on the python page make sure when downloading to add python.exe to path
# py -3.11 -m pip install mediapipe==0.10.14 opencv-python numpy

import argparse
import os
import sys
import time
 
import cv2
import mediapipe as mp
import numpy as np
 
# ── Tasks Vision Imports & Setup ─────────────────────────────────────────────
try:
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode,
    )
    from mediapipe.tasks.python.vision.pose_landmarker import (
        PoseLandmarker,
        PoseLandmarkerOptions,
    )
    from mediapipe.tasks.python.vision.face_landmarker import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        FaceLandmarksConnections,
    )
    from mediapipe.tasks.python.vision.hand_landmarker import (
        HandLandmarker,
        HandLandmarkerOptions,
        HandLandmarksConnections,
    )
    from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
    from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles

    HAS_POSE_TASKS = True
    HAS_FACE_MESH = True  # backward compatibility alias
    HAS_HANDS = True      # backward compatibility alias
except Exception as e:
    print(f"Error initializing modern Tasks API: {e}")
    HAS_POSE_TASKS = False
    HAS_FACE_MESH = False
    HAS_HANDS = False
    mp_drawing = mp_drawing_styles = None

# ── Constants ────────────────────────────────────────────────────────────────
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_FILE = "pose_landmarker_lite.task"

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_FILE = "face_landmarker.task"

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_FILE = "hand_landmarker.task"

MODEL_URL  = POSE_MODEL_URL
MODEL_FILE = POSE_MODEL_FILE
 
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

EMOJI_FONT_PATH = None
for p in [
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "/System/Library/Fonts/Supplemental/Apple Color Emoji.ttc",
    "/Library/Fonts/Apple Color Emoji.ttc"
]:
    if os.path.exists(p):
        EMOJI_FONT_PATH = p
        break
 
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
 
 
def draw_expression_label(frame, label):
    text = f"Expression: {label}"
    fw    = frame.shape[1]
    (tw,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, text, (fw-tw-10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
 
 
def get_face_bbox_from_landmarks(landmarks, img_w, img_h):
    lms = getattr(landmarks, "landmark", landmarks)
    xs = [lm.x*img_w for lm in lms]
    ys = [lm.y*img_h for lm in lms]
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
 
 
from PIL import Image, ImageDraw, ImageFont

def draw_emoji_face_pillow(frame, face_bbox, emoji_char, scale=1.0):
    x, y, w, h = face_bbox
    if w <= 0 or h <= 0:
        return
    
    target_size = int(max(w, h) * 1.4 * scale)
    if target_size < 10:
        target_size = 10
        
    base_size = 160
    
    try:
        if EMOJI_FONT_PATH:
            font = ImageFont.truetype(EMOJI_FONT_PATH, size=base_size)
        else:
            font = ImageFont.load_default()
            
        canvas = Image.new("RGBA", (base_size + 40, base_size + 40), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        try:
            bbox = draw.textbbox((0, 0), emoji_char, font=font)
            ew = bbox[2] - bbox[0]
            eh = bbox[3] - bbox[1]
            ex = (canvas.width - ew) // 2 - bbox[0]
            ey = (canvas.height - eh) // 2 - bbox[1]
        except Exception:
            ex, ey = 20, 20
            
        draw.text((ex, ey), emoji_char, font=font, embedded_color=True)
        
        resized_emoji = canvas.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cx = x + w // 2
        cy = y + h // 2
        px = cx - target_size // 2
        py = cy - target_size // 2
        
        pil_frame.paste(resized_emoji, (px, py), resized_emoji)
        cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR, dst=frame)
        
    except Exception as e:
        cv2.putText(frame, emoji_char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        print(f"Pillow drawing error: {e}")


def draw_emoji_face(image, face_bbox, sign, scale=1.0):
    # Compatibility wrapper for legacy hand gesture code
    emoji_map = {
        "fist":      "😠",
        "thumbs_up": "😉",
        "peace":     "😊",
        "surprise":  "😲",
        "neutral":   "😐",
    }
    emoji_char = emoji_map.get(sign, "😐")
    draw_emoji_face_pillow(image, face_bbox, emoji_char, scale)


def classify_expression(blendshapes):
    scores = {b.category_name: b.score for b in blendshapes}
    
    smile = (scores.get("mouthSmileLeft", 0.0) + scores.get("mouthSmileRight", 0.0)) / 2.0
    frown = (scores.get("mouthFrownLeft", 0.0) + scores.get("mouthFrownRight", 0.0)) / 2.0
    angry = (scores.get("browDownLeft", 0.0) + scores.get("browDownRight", 0.0)) / 2.0
    surprise = scores.get("jawOpen", 0.0)
    
    triggers = {
        "smile": (smile, "😊", "Happy"),
        "frown": (frown, "☹️", "Sad"),
        "angry": (angry, "😠", "Angry"),
        "surprise": (surprise, "😲", "Surprise")
    }
    
    best_cat = None
    best_score = 0.35 # threshold
    
    for cat, (score, emoji, label) in triggers.items():
        if score > best_score:
            best_score = score
            best_cat = cat
            
    if best_cat:
        return triggers[best_cat][1], triggers[best_cat][2]
    
    return "😐", "Neutral"
 
 
# ── Hand detection helpers ───────────────────────────────────────────────────
def detect_fingers(hand_landmarks):
    """Return list of finger names that are extended."""
    lm             = getattr(hand_landmarks, "landmark", hand_landmarks)
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
    return create_hand_landmarker()
 
 
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


def download_face_model():
    if os.path.exists(FACE_MODEL_FILE):
        return FACE_MODEL_FILE
    try:
        import urllib.request
        print("Downloading face model…")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_FILE)
        return FACE_MODEL_FILE
    except Exception as exc:
        print(f"Could not download face model: {exc}")
        return None


def download_hand_model():
    if os.path.exists(HAND_MODEL_FILE):
        return HAND_MODEL_FILE
    try:
        import urllib.request
        print("Downloading hand model…")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_FILE)
        return HAND_MODEL_FILE
    except Exception as exc:
        print(f"Could not download hand model: {exc}")
        return None


def create_face_landmarker():
    model_path = download_face_model()
    if not model_path:
        return None
    base_options = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )
    return FaceLandmarker.create_from_options(options)


def create_hand_landmarker():
    model_path = download_hand_model()
    if not model_path:
        return None
    base_options = BaseOptions(model_asset_path=model_path)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)
 
 
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
 
    face_landmarker = None
    if HAS_FACE_MESH:
        face_landmarker = create_face_landmarker()
    else:
        print("Face landmarker unavailable; using OpenCV fallback.")
 
    frame_index = 0
    prev_time   = time.time()
    last_saved  = False
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame     = cv2.flip(frame, 1)
        annotated = frame.copy()
 
        if face_landmarker:
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_index * 1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 30))
            result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            if result.face_landmarks:
                for fl in result.face_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, fl, FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                        None, mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        annotated, fl, FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                        None, mp_drawing_styles.get_default_face_mesh_contours_style())
                    
                    emoji_char = "😐"
                    expression = "Neutral"
                    if result.face_blendshapes:
                        emoji_char, expression = classify_expression(result.face_blendshapes[0])
                    
                    face_bbox = get_face_bbox_from_landmarks(fl, frame.shape[1], frame.shape[0])
                    draw_emoji_face_pillow(annotated, face_bbox, emoji_char, scale=1.0)
        else:
            faces = detect_faces_opencv(frame)
            for face in faces:
                draw_emoji_face_pillow(annotated, face, "😐", scale=1.0)
 
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
 
    if face_landmarker: face_landmarker.close()
    cap.release(); cv2.destroyAllWindows()
    return 0
 
 
def run_finger_tracking(source, backend, show_gui, headless, output_path, frame_limit=0):
    if not HAS_HANDS:
        print("Finger tracking requires MediaPipe HandLandmarker."); return 1
    cap = open_capture(source, backend)
    if not cap.isOpened():
        print(f"Could not open: {source}"); return 1
 
    WIN   = "Finger Tracking"
    hands = create_hand_landmarker()
    if hands is None:
        cap.release(); return 1
        
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_index * 1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 30))
        result = hands.detect_for_video(mp_image, timestamp_ms)
 
        annotated      = frame.copy()
        active_fingers = []
 
        if result.hand_landmarks:
            for hl in result.hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, hl, HandLandmarksConnections.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
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
    cap = open_capture(source, backend)
    if not cap.isOpened():
        print(f"Could not open: {source}"); return 1
 
    pose_landmarker = create_pose_landmarker() if HAS_POSE_TASKS else None
    face_landmarker = create_face_landmarker() if HAS_FACE_MESH else None
    hand_landmarker = create_hand_landmarker() if HAS_HANDS else None
 
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
            timestamp_ms = int(frame_index * 1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 30))
            result   = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if getattr(result,"pose_landmarks",None):
                body_pose_landmarks = result.pose_landmarks[0]
                for pl in result.pose_landmarks:
                    draw_pose_annotations(annotated, pl)
 
        # ── Flip for hand + face (mirror = natural) ──
        flip           = cv2.flip(frame, 1)
        active_fingers = []
        emoji_char     = "😐"
        expression     = "Neutral"
 
        # ── Hand detection ──
        if hand_landmarker is not None:
            rgb_h = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
            mp_image_h = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_h)
            timestamp_ms = int(frame_index * 1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 30))
            hand_results = hand_landmarker.detect_for_video(mp_image_h, timestamp_ms)
 
            if hand_results.hand_landmarks:
                for hl in hand_results.hand_landmarks:
                    mp_drawing.draw_landmarks(
                        flip, hl, HandLandmarksConnections.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    active_fingers.extend(detect_fingers(hl))
 
        # ── Face emoji driven by facial expressions ──
        if face_landmarker is not None:
            rgb_f    = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
            mp_image_f = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_f)
            timestamp_ms = int(frame_index * 1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 30))
            f_result = face_landmarker.detect_for_video(mp_image_f, timestamp_ms)
            if f_result.face_landmarks:
                fl         = f_result.face_landmarks[0]
                face_bbox  = get_face_bbox_from_landmarks(fl, flip.shape[1], flip.shape[0])
                emoji_scale= estimate_emoji_scale(face_bbox, body_pose_landmarks,
                                                  flip.shape[1], flip.shape[0])
                if f_result.face_blendshapes:
                    emoji_char, expression = classify_expression(f_result.face_blendshapes[0])
                draw_emoji_face_pillow(flip, face_bbox, emoji_char, emoji_scale)
            annotated = cv2.flip(flip, 1)
        else:
            faces = detect_faces_opencv(flip)
            for face in faces:
                emoji_scale = estimate_emoji_scale(face, body_pose_landmarks,
                                                   flip.shape[1], flip.shape[0])
                draw_emoji_face_pillow(flip, face, "😐", emoji_scale)
            annotated = cv2.flip(flip, 1)
 
        # ── HUD ──
        cv2.putText(annotated, f"Fingers Up: {len(active_fingers)}", (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        if active_fingers:
            cv2.putText(annotated, ", ".join(dict.fromkeys(active_fingers)), (30,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
 
        draw_expression_label(annotated, expression)
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
    if face_landmarker: face_landmarker.close()
    if hand_landmarker: hand_landmarker.close()
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
