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
def draw_pose_annotations(frame, pose_landmarks, mirrored=False):
    h, w = frame.shape[:2]
    if mirrored:
        pts  = [(int((1.0 - lm.x)*w), int(lm.y*h)) for lm in pose_landmarks]
    else:
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
    cv2.putText(frame, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
 
 
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


def draw_text_with_emoji(canvas, text, emoji, x, y, font_size=14, color=(200, 200, 200)):
    # 1. Draw text with OpenCV
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    
    # 2. Draw emoji using Pillow next to text
    emoji_x = x + tw + 6
    emoji_y = y - th - 2
    
    base_size = 32
    try:
        if EMOJI_FONT_PATH:
            font = ImageFont.truetype(EMOJI_FONT_PATH, size=base_size)
        else:
            font = ImageFont.load_default()
        
        # Create transparent canvas for emoji
        emoji_canvas = Image.new("RGBA", (base_size + 10, base_size + 10), (0, 0, 0, 0))
        draw = ImageDraw.Draw(emoji_canvas)
        draw.text((5, 5), emoji, font=font, embedded_color=True)
        
        # Resize to target size
        target_size = font_size + 4
        resized = emoji_canvas.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Crop canvas region and convert to PIL
        h, w = canvas.shape[:2]
        rx = min(w - target_size, max(0, emoji_x))
        ry = min(h - target_size, max(0, emoji_y))
        
        roi = canvas[ry:ry+target_size, rx:rx+target_size]
        pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        pil_roi.paste(resized, (0, 0), resized)
        
        canvas[ry:ry+target_size, rx:rx+target_size] = cv2.cvtColor(np.array(pil_roi), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error drawing label emoji: {e}")


def draw_right_panel(frame, album_faces, current_expression):
    # original frame size
    H, W = frame.shape[:2]
    panel_w = 200
    
    # Create combined canvas of size (H, W + panel_w)
    canvas = np.zeros((H, W + panel_w, 3), dtype=np.uint8)
    # Copy original frame to the left side
    canvas[:, :W] = frame
    # Color the right panel dark gray
    canvas[:, W:] = (30, 30, 30) # Dark gray background
    
    # 4 Slots
    slot_h = H // 4
    expressions = [
        ("Happy", "😊", "happy"),
        ("Sad", "☹️", "sad"),
        ("Angry", "😠", "angry"),
        ("Surprise", "😲", "surprise")
    ]
    
    # Dynamic box sizing to prevent out-of-bounds on small frames
    box_h = min(80, int(slot_h * 0.7))
    box_w = box_h
    
    for idx, (label, emoji, key) in enumerate(expressions):
        # Calculate slot y-bounds
        y_start = idx * slot_h
        
        # Center of slot: (W + panel_w // 2, y_start + slot_h // 2)
        bx = W + (panel_w - box_w) // 2
        by = y_start + (slot_h - box_h) // 2
        
        # Draw slot header / label using the helper function
        th = max(8, int(slot_h * 0.12))
        draw_text_with_emoji(canvas, label, emoji, W + 15, y_start + th + 10, font_size=12)
        
        # Draw face crop if available, otherwise placeholder
        if key in album_faces:
            crop = album_faces[key]
            if crop is not None and crop.size > 0:
                crop_resized = cv2.resize(crop, (box_w, box_h))
                canvas[by:by+box_h, bx:bx+box_w] = crop_resized
                # Draw green border if it's the active one
                border_color = (0, 255, 0) if current_expression == label else (150, 150, 150)
                cv2.rectangle(canvas, (bx-1, by-1), (bx+box_w, by+box_h), border_color, 2)
        else:
            # Placeholder: gray square with a dashed or thin border
            cv2.rectangle(canvas, (bx, by), (bx+box_w, by+box_h), (80, 80, 80), 1)
            # Draw question mark or instructions
            q_scale = max(0.5, box_h / 80.0)
            cv2.putText(canvas, "?", (bx + int(box_w * 0.35), by + int(box_h * 0.65)), 
                        cv2.FONT_HERSHEY_SIMPLEX, q_scale, (100, 100, 100), 2)
            
    return canvas


BOOTH_STEPS = {
    0: ("Happy", "happy", "Challenge 1: Make a HAPPY face! 😊"),
    1: ("Sad", "sad", "Challenge 2: Make a SAD face! ☹️"),
    2: ("Angry", "angry", "Challenge 3: Make an ANGRY face! 😠"),
    3: ("Surprise", "surprise", "Challenge 4: Make a SURPRISE face! 😲")
}

def draw_instructions_banner(frame, step_idx, flash_frames):
    H, W = frame.shape[:2]
    banner_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - banner_h), (W, H), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, dst=frame)
    
    if step_idx < 4:
        _, _, prompt = BOOTH_STEPS[step_idx]
        text = prompt
        color = (50, 255, 50)
    elif step_idx == 4:
        text = "All expressions captured! Press ENTER to view your album."
        color = (0, 255, 255)
    else:
        text = "Album Complete! Press 'r' to reset or 'q' to quit."
        color = (0, 255, 255)
        
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    tx = (W - tw) // 2
    ty = H - (banner_h - th) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    
    if flash_frames > 0:
        flash_overlay = np.ones((H, W, 3), dtype=np.uint8) * 255
        alpha = flash_frames * 0.18
        cv2.addWeighted(flash_overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)


def draw_album_view(H, W_total, album_faces):
    # Create background canvas
    canvas = np.zeros((H, W_total, 3), dtype=np.uint8)
    canvas[:] = (30, 28, 28) # Sleek dark slate
    
    # Title
    title = "YOUR EXPRESSION ALBUM"
    title_scale = max(0.55, min(0.75, H / 640.0))
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 2)
    tx = (W_total - tw) // 2
    cv2.putText(canvas, title, (tx, int(H * 0.12)), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 255, 255), 2, cv2.LINE_AA)
    
    expressions = [
        ("Happy", "😊", "happy"),
        ("Sad", "☹️", "sad"),
        ("Angry", "😠", "angry"),
        ("Surprise", "😲", "surprise")
    ]
    
    # Determine card size dynamically
    card_w = min(160, int(W_total * 0.18))
    card_h = int(card_w * 1.35)
    
    # Spacing between cards:
    spacing = (W_total - 4 * card_w) // 5
    card_y = (H - card_h) // 2 + int(H * 0.05)
    
    for idx, (label, emoji, key) in enumerate(expressions):
        card_x = spacing + idx * (card_w + spacing)
        
        # Draw Polaroid background (white/cream card)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (245, 245, 245), -1)
        # Draw thin gray border for the card
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (200, 200, 200), 1)
        
        # Photo area size: 85% of card width
        photo_margin = int(card_w * 0.06)
        photo_x = card_x + photo_margin
        photo_y = card_y + photo_margin
        photo_w = card_w - 2 * photo_margin
        photo_h = photo_w
        
        # Get crop or placeholder
        if key in album_faces:
            crop = album_faces[key]
            # Ensure crop has content
            if crop is not None and crop.size > 0:
                crop_resized = cv2.resize(crop, (photo_w, photo_h))
                canvas[photo_y:photo_y+photo_h, photo_x:photo_x+photo_w] = crop_resized
            else:
                # Black fallback
                cv2.rectangle(canvas, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h), (0, 0, 0), -1)
        else:
            # Gray placeholder
            cv2.rectangle(canvas, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h), (200, 200, 200), -1)
            cv2.putText(canvas, "?", (photo_x + int(photo_w * 0.35), photo_y + int(photo_h * 0.65)), 
                        cv2.FONT_HERSHEY_SIMPLEX, photo_w / 80.0, (150, 150, 150), 2)
            
        # Draw photo frame border
        cv2.rectangle(canvas, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h), (180, 180, 180), 1)
        
        # Draw label text below photo
        text_y = card_y + photo_margin + photo_h + int(card_h * 0.15)
        font_sz = max(10, int(card_w * 0.08))
        draw_text_with_emoji(canvas, label, emoji, card_x + photo_margin, text_y, font_size=font_sz, color=(40, 40, 40))
        
    # Draw instructions footer
    footer = "Press 'r' to Restart / Retake  |  Press 'q' to Quit"
    footer_scale = max(0.35, min(0.45, H / 1000.0))
    (ftw, fth), _ = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, footer_scale, 1)
    ftx = (W_total - ftw) // 2
    cv2.putText(canvas, footer, (ftx, H - 25), cv2.FONT_HERSHEY_SIMPLEX, footer_scale, (180, 180, 180), 1, cv2.LINE_AA)
    
    return canvas



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
    album_faces = {}
    
    # Booth game states
    booth_state = 0
    flash_frames = 0
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame     = cv2.flip(frame, 1)
        
        if flash_frames > 0:
            flash_frames -= 1
 
        if booth_state == 5:
            # Render Polaroid Album collage
            annotated = draw_album_view(frame.shape[0], frame.shape[1] + 200, album_faces)
            if headless:
                cv2.imwrite(output_path, annotated)
                print(f"Saved completed album to {output_path}")
                break
        else:
            annotated = frame.copy()
            expression = "Neutral"
            
            # Headless transition from state 4 to state 5
            if headless and booth_state == 4:
                booth_state = 5
                annotated = draw_album_view(frame.shape[0], frame.shape[1] + 200, album_faces)
                cv2.imwrite(output_path, annotated)
                print(f"Saved completed album to {output_path}")
                break
 
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
                        if result.face_blendshapes:
                            emoji_char, expression = classify_expression(result.face_blendshapes[0])
                        
                        face_bbox = get_face_bbox_from_landmarks(fl, frame.shape[1], frame.shape[0])
                        
                        # Capture face crop from the clean frame before overlays are drawn
                        x, y, w_box, h_box = face_bbox
                        img_h, img_w = frame.shape[:2]
                        pad_x = int(w_box * 0.15)
                        pad_y = int(h_box * 0.2)
                        x0 = max(0, x - pad_x)
                        y0 = max(0, y - pad_y)
                        x1 = min(img_w, x + w_box + pad_x)
                        y1 = min(img_h, y + h_box + pad_y)
                        
                        if (x1 - x0) > 10 and (y1 - y0) > 10:
                            face_crop = frame[y0:y1, x0:x1].copy()
                            
                            # Sequential capture flow
                            if booth_state < 4:
                                target_label, key, prompt = BOOTH_STEPS[booth_state]
                                if expression == target_label or (headless and frame_index >= booth_state * 2):
                                    album_faces[key] = face_crop
                                    cv2.imwrite(f"{key}.png", face_crop)
                                    flash_frames = 5
                                    booth_state += 1
                        
                        draw_emoji_face_pillow(annotated, face_bbox, emoji_char, scale=1.0)
                elif headless and booth_state < 4:
                    # Fallback crop for headless tests without landmarks
                    target_label, key, prompt = BOOTH_STEPS[booth_state]
                    face_crop = np.zeros((100, 100, 3), dtype=np.uint8)
                    album_faces[key] = face_crop
                    cv2.imwrite(f"{key}.png", face_crop)
                    flash_frames = 5
                    booth_state += 1
            elif headless and booth_state < 4:
                # Fallback crop for headless tests without face landmarker
                target_label, key, prompt = BOOTH_STEPS[booth_state]
                face_crop = np.zeros((100, 100, 3), dtype=np.uint8)
                album_faces[key] = face_crop
                cv2.imwrite(f"{key}.png", face_crop)
                flash_frames = 5
                booth_state += 1
                
            # Draw HUD & Instructions Banner first on annotated
            draw_instructions_banner(annotated, booth_state, flash_frames)
            
            fps = 1.0/(time.time()-prev_time) if time.time()>prev_time else 0.0
            prev_time = time.time()
            draw_info(annotated, "Face", fps)
            
            # ── Draw right panel last ──
            current_active = BOOTH_STEPS[booth_state][0] if (booth_state < 4) else "Neutral"
            annotated = draw_right_panel(annotated, album_faces, current_active)
            
        frame_index += 1
 
        if headless:
            if frame_limit and frame_index >= frame_limit:
                cv2.imwrite(output_path, annotated)
                print(f"Saved frame {frame_index} to {output_path}")
                break
            continue
            
        if show_gui and is_gui_available():
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("f"): toggle_fullscreen(WIN)
            if key == ord("r"):
                album_faces.clear()
                booth_state = 0
                flash_frames = 0
            if key in (13, 10, 32) and booth_state == 4:
                booth_state = 5
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
 
        cv2.putText(annotated, f"Fingers Up: {len(active_fingers)}", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        if active_fingers:
            cv2.putText(annotated, ", ".join(active_fingers), (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
 
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
    album_faces = {}
    
    # Booth game states
    booth_state = 0
    flash_frames = 0
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
 
        # ── Flip for hand + face + pose (mirror = natural) ──
        flip           = cv2.flip(frame, 1)
        
        if flash_frames > 0:
            flash_frames -= 1
            
        if booth_state == 5:
            # Render Polaroid Album collage
            annotated = draw_album_view(flip.shape[0], flip.shape[1] + 200, album_faces)
            if headless:
                cv2.imwrite(output_path, annotated)
                print(f"Saved completed album to {output_path}")
                break
        else:
            active_fingers = []
            emoji_char     = "😐"
            expression     = "Neutral"
            body_pose_landmarks = None
            
            # Headless transition from state 4 to state 5
            if headless and booth_state == 4:
                booth_state = 5
                annotated = draw_album_view(flip.shape[0], flip.shape[1] + 200, album_faces)
                cv2.imwrite(output_path, annotated)
                print(f"Saved completed album to {output_path}")
                break
                
            # ── Pose on mirrored frame ──
            if pose_landmarker is not None:
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_index * 1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 30))
                result   = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                if getattr(result,"pose_landmarks",None):
                    body_pose_landmarks = result.pose_landmarks[0]
                    for pl in result.pose_landmarks:
                        draw_pose_annotations(flip, pl, mirrored=True)
 
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
                    
                    # Capture face crop from the clean frame before overlays are drawn
                    x, y, w_box, h_box = face_bbox
                    img_h, img_w = flip.shape[:2]
                    pad_x = int(w_box * 0.15)
                    pad_y = int(h_box * 0.2)
                    x0 = max(0, x - pad_x)
                    y0 = max(0, y - pad_y)
                    x1 = min(img_w, x + w_box + pad_x)
                    y1 = min(img_h, y + h_box + pad_y)
                    
                    if (x1 - x0) > 10 and (y1 - y0) > 10:
                        face_crop = flip[y0:y1, x0:x1].copy()
                        
                        # Sequential capture flow
                        if booth_state < 4:
                            target_label, key, prompt = BOOTH_STEPS[booth_state]
                            if expression == target_label or (headless and frame_index >= booth_state * 2):
                                album_faces[key] = face_crop
                                cv2.imwrite(f"{key}.png", face_crop)
                                flash_frames = 5
                                booth_state += 1
                    
                    draw_emoji_face_pillow(flip, face_bbox, emoji_char, emoji_scale)
                elif headless and booth_state < 4:
                    # Fallback crop for headless tests without landmarks
                    target_label, key, prompt = BOOTH_STEPS[booth_state]
                    face_crop = np.zeros((100, 100, 3), dtype=np.uint8)
                    album_faces[key] = face_crop
                    cv2.imwrite(f"{key}.png", face_crop)
                    flash_frames = 5
                    booth_state += 1
            elif headless and booth_state < 4:
                # Fallback crop for headless tests without face landmarker
                target_label, key, prompt = BOOTH_STEPS[booth_state]
                face_crop = np.zeros((100, 100, 3), dtype=np.uint8)
                album_faces[key] = face_crop
                cv2.imwrite(f"{key}.png", face_crop)
                flash_frames = 5
                booth_state += 1
 
            # Draw HUD & Instructions Banner first on flip
            draw_instructions_banner(flip, booth_state, flash_frames)
 
            # ── HUD ──
            cv2.putText(flip, f"Fingers Up: {len(active_fingers)}", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
            if active_fingers:
                cv2.putText(flip, ", ".join(dict.fromkeys(active_fingers)), (10,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
 
            draw_expression_label(flip, expression)
            fps = 1.0/(time.time()-prev_time) if time.time()>prev_time else 0.0
            prev_time = time.time()
            draw_info(flip, "All", fps)
 
            # ── Draw right panel last ──
            current_active = BOOTH_STEPS[booth_state][0] if (booth_state < 4) else "Neutral"
            annotated = draw_right_panel(flip, album_faces, current_active)
            
        frame_index += 1
 
        if headless:
            if frame_limit and frame_index >= frame_limit:
                cv2.imwrite(output_path, annotated)
                print(f"Saved frame {frame_index} to {output_path}")
                break
            continue
 
        if show_gui and is_gui_available():
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("f"): toggle_fullscreen(WIN)
            if key == ord("r"):
                album_faces.clear()
                booth_state = 0
                flash_frames = 0
            if key in (13, 10, 32) and booth_state == 4:
                booth_state = 5
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
