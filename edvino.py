import argparse
import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import io
import time
from http import server
import socketserver

# Initialize MediaPipe Face Mesh
# Try to access the legacy `solutions` API; newer mediapipe releases expose
# a different API surface (tasks). If `mp.solutions` is missing, we'll
# fall back to a limited synthetic-mode so the script can still run for tests.
try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    HAS_MEDIAPIPE_SOLUTIONS = True
except Exception:
    mp_face_mesh = None
    mp_drawing = None
    mp_drawing_styles = None
    HAS_MEDIAPIPE_SOLUTIONS = False

# Detect newer MediaPipe Tasks API (face_landmarker) if available.
# We'll try to import the minimal bits and expose a factory that may
# be used elsewhere. If not available, we silently fall back to
# OpenCV Haar cascade detection implemented below.
HAS_MEDIAPIPE_TASKS = False
FaceLandmarker = None
_mp_tasks_image = None
try:
    # Import the Task-based face landmarker if present
    from mediapipe.tasks.python.vision.face_landmarker import (
        FaceLandmarker as _FaceLandmarkerClass,
    )
    # lightweight image helper module (may or may not exist depending on version)
    try:
        from mediapipe.tasks.python.vision.core import image as _mp_tasks_image
    except Exception:
        _mp_tasks_image = None

    FaceLandmarker = _FaceLandmarkerClass
    HAS_MEDIAPIPE_TASKS = True
except Exception:
    HAS_MEDIAPIPE_TASKS = False
    FaceLandmarker = None
    _mp_tasks_image = None


def parse_camera_arg(camera_arg):
    if camera_arg is None:
        return 0
    if isinstance(camera_arg, int):
        return camera_arg
    if str(camera_arg).isdigit():
        return int(camera_arg)
    return camera_arg


def can_show_gui():
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return None
    return detector


FACE_DETECTOR = load_face_detector()


def detect_faces_opencv(frame):
    if FACE_DETECTOR is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces


# ----- Mediapipe Tasks helper (best-effort) -----
def create_task_landmarker_from_asset(asset_name='face_landmarker_v2.task'):
    """Try to create a FaceLandmarker from the given asset name.
    Returns the landmarker instance or None on failure.
    """
    if not HAS_MEDIAPIPE_TASKS or FaceLandmarker is None:
        return None
    try:
        # Try convenience creation from bundled asset name. If the asset
        # is not available in the environment, this will raise and we
        # gracefully fall back to OpenCV detection.
        lm = FaceLandmarker.create_from_model_path(asset_name)
        return lm
    except Exception:
        return None


def detect_with_task_landmarker(landmarker, frame):
    """Run task-based landmarker on a BGR numpy frame and return
    a list of face landmarks-like objects (or empty list).
    This uses a temporary file as the task API expects an Image.
    """
    if landmarker is None:
        return []
    if _mp_tasks_image is None:
        return []
    import tempfile
    try:
        # Encode frame to JPEG and write to a temp file
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            return []
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(buf.tobytes())
            tmp_path = f.name
        img = _mp_tasks_image.Image.create_from_file(tmp_path)
        # For simple usage we call detect (static image)
        res = landmarker.detect(img)
        # res may have .face_landmarks or similar; map to a simple list
        faces = []
        if hasattr(res, 'face_landmarks') and res.face_landmarks:
            faces = res.face_landmarks
        elif hasattr(res, 'landmarks') and res.landmarks:
            faces = res.landmarks
        # best-effort: return whatever landmarks we got
        return faces
    except Exception:
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def draw_face_overlay(image, faces):
    for (x, y, w, h) in faces:
        # draw rectangle and emoji-style eyes/mouth within the face region
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


def list_cameras(max_devices=8):
    """Return a list of available camera indices (best-effort)."""
    found = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        if ret:
            found.append(idx)
        cap.release()
    return found


def start_mjpeg_stream(cap, port, processed=False):
    """Start a simple MJPEG HTTP server that serves frames from `cap`.
    If `processed` is True, the frames will include the emoji overlay (dummy),
    otherwise the raw frames are served.
    """
    if port <= 0:
        raise ValueError("port must be > 0")

    frame_lock = threading.Lock()
    latest = {'jpg': None}

    def reader():
        # If processed == 'mediapipe' or processed == 'auto' we will prefer
        # the Task-based FaceLandmarker (if available), then fall back to
        # legacy mp.solutions FaceMesh, then OpenCV Haar cascade.
        local_task_landmarker = None
        local_face_mesh = None
        if processed in ('mediapipe', 'auto'):
            # Try task API first
            try:
                local_task_landmarker = create_task_landmarker_from_asset('face_landmarker_v2.task')
            except Exception:
                local_task_landmarker = None
            # If tasks landmarker not available, try legacy solutions API
            if local_task_landmarker is None and mp_face_mesh is not None:
                try:
                    local_face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                except Exception:
                    local_face_mesh = None

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            # Use Task API landmarker if we created one
            if local_task_landmarker is not None:
                try:
                    faces = detect_with_task_landmarker(local_task_landmarker, frame)
                except Exception:
                    faces = []
                emoji = np.zeros_like(frame)
                if faces:
                    for face_landmarks in faces:
                        draw_emoji_eyes(emoji, face_landmarks, None)
                        draw_emoji_mouth(emoji, face_landmarks, None)
                blended = cv2.addWeighted(frame, 0.6, emoji, 0.4, 0)
                out = blended
            elif processed == 'mediapipe' and local_face_mesh is not None:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = local_face_mesh.process(rgb)
                emoji = np.zeros_like(frame)
                if results and getattr(results, 'multi_face_landmarks', None):
                    for face_landmarks in results.multi_face_landmarks:
                        draw_emoji_eyes(emoji, face_landmarks, None)
                        draw_emoji_mouth(emoji, face_landmarks, None)
                blended = cv2.addWeighted(frame, 0.6, emoji, 0.4, 0)
                out = blended
            elif processed:
                faces = detect_faces_opencv(frame)
                emoji = np.zeros_like(frame)
                draw_face_overlay(emoji, faces)
                blended = cv2.addWeighted(frame, 0.6, emoji, 0.4, 0)
                out = blended
            else:
                out = frame

            ret2, jpg = cv2.imencode('.jpg', out)
            if not ret2:
                continue
            with frame_lock:
                latest['jpg'] = jpg.tobytes()

    class MJPEGHandler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/':
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        jpg = latest['jpg']
                    if jpg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.03)
            except Exception:
                return

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, server.HTTPServer):
        allow_reuse_address = True

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    httpd = ThreadingHTTPServer(('0.0.0.0', port), MJPEGHandler)
    print(f"MJPEG stream available at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()

# --- Custom Emoji-like Drawing Functions (Simplified) ---
# These functions will draw exaggerated features onto the face.
# For full emoji merging, you would use actual image assets and complex transformations.

def draw_emoji_eyes(image, landmarks, face_2d_points):
    """
    Draws simplified emoji-like eyes on the face.
    We'll pick a few key landmarks for eye centers.
    """
    # Landmark indices for eyes (approximate centers for simplification)
    # These are just examples, you might want to refine them.
    left_eye_center_idx = 1 # Example landmark for left eye center
    right_eye_center_idx = 4 # Example landmark for right eye center

    if landmarks:
        img_h, img_w, _ = image.shape
        left_eye_x = int(landmarks.landmark[left_eye_center_idx].x * img_w)
        left_eye_y = int(landmarks.landmark[left_eye_center_idx].y * img_h)
        right_eye_x = int(landmarks.landmark[right_eye_center_idx].x * img_w)
        right_eye_y = int(landmarks.landmark[right_eye_center_idx].y * img_h)

        # Draw large, cartoonish eyes (e.g., white circles with black pupils)
        # Left eye
        cv2.circle(image, (left_eye_x, left_eye_y), 30, (255, 255, 255), -1) # White outer
        cv2.circle(image, (left_eye_x, left_eye_y), 15, (0, 0, 0), -1)       # Black pupil
        # Right eye
        cv2.circle(image, (right_eye_x, right_eye_y), 30, (255, 255, 255), -1) # White outer
        cv2.circle(image, (right_eye_x, right_eye_y), 15, (0, 0, 0), -1)       # Black pupil

def draw_emoji_mouth(image, landmarks, face_2d_points):
    """
    Draws a simplified emoji-like mouth on the face.
    """
    # Landmark indices for mouth (approximate corners and center)
    mouth_left_idx = 78
    mouth_right_idx = 308
    mouth_top_idx = 13
    mouth_bottom_idx = 14

    if landmarks:
        img_h, img_w, _ = image.shape
        mouth_points = []
        for idx in [mouth_left_idx, mouth_top_idx, mouth_right_idx, mouth_bottom_idx]:
             mouth_points.append([int(landmarks.landmark[idx].x * img_w),
                                  int(landmarks.landmark[idx].y * img_h)])
        mouth_points = np.array(mouth_points, np.int32).reshape((-1, 1, 2))

        # Draw a simple red mouth shape
        cv2.polylines(image, [mouth_points], True, (0, 0, 255), 5) # Red outline
        cv2.fillPoly(image, [mouth_points], (0, 0, 200)) # Slightly darker red fill


def get_head_pose(landmarks, img_w, img_h):
    """
    Estimates head pose (rotation) from face landmarks.
    This is a simplification; for accurate 3D pose, you'd use Perspective-n-Point.
    But for simple rotation, relative landmark positions can indicate it.
    """
    if not landmarks:
        return 0, 0, 0 # Returns pitch, yaw, roll (simplified)

    # Key points for nose, left eye corner, right eye corner, mouth corners
    # These are approximation for demonstration.
    nose_tip = (landmarks.landmark[1].x * img_w, landmarks.landmark[1].y * img_h)
    nose_bottom = (landmarks.landmark[2].x * img_w, landmarks.landmark[2].y * img_h)
    left_eye = (landmarks.landmark[226].x * img_w, landmarks.landmark[226].y * img_h)
    right_eye = (landmarks.landmark[446].x * img_w, landmarks.landmark[446].y * img_h)
    mouth_left = (landmarks.landmark[78].x * img_w, landmarks.landmark[78].y * img_h)
    mouth_right = (landmarks.landmark[308].x * img_w, landmarks.landmark[308].y * img_h)

    # Simple horizontal offset to estimate yaw
    avg_eye_x = (left_eye[0] + right_eye[0]) / 2
    yaw_factor = (nose_tip[0] - avg_eye_x) / img_w # Positive if looking right, negative if looking left
    yaw = yaw_factor * 60 # Scale to a reasonable angle

    # Simple vertical offset to estimate pitch (looking up/down)
    pitch_factor = (nose_tip[1] - ((left_eye[1] + right_eye[1]) / 2)) / img_h
    pitch = pitch_factor * 40 # Scale to a reasonable angle

    # Simpler roll: Difference in y-coords of mouth corners
    roll_factor = (mouth_left[1] - mouth_right[1]) / (abs(mouth_left[0] - mouth_right[0]) + 1e-6)
    roll = roll_factor * 30 # Scale to a reasonable angle

    return pitch, yaw, roll


class _SimpleLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def make_dummy_landmarks(norm_x, norm_y, total=500):
    """Create a simple dummy landmarks object compatible with the drawing
    helpers used in this file. Landmarks are normalized (0..1).
    """
    class L:
        pass

    obj = L()
    lm = [_SimpleLandmark(0.5, 0.5) for _ in range(total)]
    # provide some common indices used by the drawing code
    idxs = [1, 2, 4, 13, 14, 78, 226, 308, 446]
    for i in idxs:
        if i < total:
            lm[i] = _SimpleLandmark(norm_x, norm_y)
    obj.landmark = lm
    return obj

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Emoji-style face overlay (camera or synthetic)")
    parser.add_argument("--camera", default=None, help="Camera index or device path (e.g. 0 or /dev/video0)")
    parser.add_argument("--input", default=None, help="Video file input path (overrides --camera)")
    parser.add_argument("--raw", action="store_true", help="If mediapipe missing, show raw camera preview instead of synthetic test")
    parser.add_argument("--backend", default="auto", choices=["auto","v4l2","dshow","avfoundation","gstreamer"],
                        help="Capture backend to use (auto chooses by OS).")
    parser.add_argument("--show-gui", action="store_true", help="Show OpenCV GUI windows if available")
    parser.add_argument("--mjpeg-port", type=int, default=0, help="If set, serve MJPEG stream on this port so you can view camera in a browser")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera indices and exit")
    args = parser.parse_args()

    if args.list_cameras:
        cams = list_cameras(max_devices=16)
        if not cams:
            print("No cameras detected (tried indices 0..15).")
        else:
            print("Detected camera indices:")
            for c in cams:
                print(f"  {c}")
            print("Use --camera INDEX to open a specific device.")
        return

    if not (HAS_MEDIAPIPE_SOLUTIONS or HAS_MEDIAPIPE_TASKS):
        # Run a short synthetic, non-GUI test so the script can be validated
        # without any available MediaPipe API (neither legacy solutions nor
        # the newer tasks API).
        print("mediapipe.solutions not available — running synthetic headless test.")
        h, w = 480, 640
        # If the user specified a camera, try to open it and show a preview
        def choose_api(backend_str):
            # Map friendly names to OpenCV apiPreference constants
            mapping = {
                'v4l2': getattr(cv2, 'CAP_V4L2', 0),
                'dshow': getattr(cv2, 'CAP_DSHOW', 0),
                'avfoundation': getattr(cv2, 'CAP_AVFOUNDATION', 0),
                'gstreamer': getattr(cv2, 'CAP_GSTREAMER', 0),
            }
            return mapping.get(backend_str, 0)

        api_pref = 0
        if args.backend != 'auto':
            api_pref = choose_api(args.backend)
        else:
            import platform
            sys_plat = platform.system().lower()
            if 'darwin' in sys_plat:
                api_pref = choose_api('avfoundation')
            elif 'windows' in sys_plat:
                api_pref = choose_api('dshow')
            else:
                api_pref = choose_api('v4l2')

        if args.camera is not None:
            cam = parse_camera_arg(args.camera)
            print(f"Attempting to open camera {cam} with api preference {api_pref}")
            cap = cv2.VideoCapture(cam, api_pref) if api_pref else cv2.VideoCapture(cam)
            if not cap.isOpened():
                print(f"Could not open camera/device: {args.camera}")
            else:
                # If MJPEG requested, stream processed frames to browser
                if args.mjpeg_port:
                    print(f"Starting MJPEG stream on port {args.mjpeg_port}")
                    start_mjpeg_stream(cap, args.mjpeg_port, processed='auto')
                    return
                if args.show_gui and can_show_gui():
                    print("Showing camera preview with face detection overlay. Press 'q' to quit.")
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        faces = detect_faces_opencv(frame)
                        emoji_frame = np.zeros_like(frame)
                        draw_face_overlay(emoji_frame, faces)
                        cv2.imshow('Raw Camera (Original)', frame)
                        cv2.imshow('Raw Camera (Face Detection Overlay)', emoji_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                print("No GUI display enabled. Use --show-gui on a machine with display or --mjpeg-port PORT to view in a browser.")
                cap.release()
                return

        # If no camera specified or camera couldn't be opened, show synthetic frames (no saving)
        if args.show_gui and can_show_gui():
            print("No camera opened — showing synthetic demo frames in a window for 30 frames.")
            for i in range(30):
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                cx = int(w/2 + 100 * math.sin(i / 6.0))
                cy = int(h/2 + 30 * math.cos(i / 8.0))
                cv2.circle(frame, (cx, cy), 60, (255, 255, 255), -1)

                emoji_frame = np.zeros_like(frame)
                dummy = make_dummy_landmarks(norm_x=cx / w, norm_y=cy / h)
                draw_emoji_eyes(emoji_frame, dummy, None)
                draw_emoji_mouth(emoji_frame, dummy, None)

                cv2.imshow('Synthetic Demo - Original', frame)
                cv2.imshow('Synthetic Demo - Emoji', emoji_frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        else:
            print("No GUI display enabled. Running synthetic demo headlessly.")
            for i in range(30):
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                cx = int(w/2 + 100 * math.sin(i / 6.0))
                cy = int(h/2 + 30 * math.cos(i / 8.0))
                cv2.circle(frame, (cx, cy), 60, (255, 255, 255), -1)
                emoji_frame = np.zeros_like(frame)
                dummy = make_dummy_landmarks(norm_x=cx / w, norm_y=cy / h)
                draw_emoji_eyes(emoji_frame, dummy, None)
                draw_emoji_mouth(emoji_frame, dummy, None)
            print("Synthetic demo completed headlessly.")
        return

    # Decide input source: video file, device path, or camera index
    source = None
    if args.input:
        source = args.input
    elif args.camera is not None:
        source = parse_camera_arg(args.camera)
    else:
        source = 0

    # Capture webcam/feed
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Warning: Could not open video source: {source}")
        # If the user didn't explicitly request a camera or input file,
        # fall back to synthetic demo so the script remains runnable.
        if args.input is None and args.camera is None:
            print("No camera available — running synthetic demo (30 frames).")
            h, w = 480, 640
            for i in range(30):
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                cx = int(w/2 + 100 * math.sin(i / 6.0))
                cy = int(h/2 + 30 * math.cos(i / 8.0))
                cv2.circle(frame, (cx, cy), 60, (255, 255, 255), -1)
                emoji_frame = np.zeros_like(frame)
                dummy = make_dummy_landmarks(norm_x=cx / w, norm_y=cy / h)
                draw_emoji_eyes(emoji_frame, dummy, None)
                draw_emoji_mouth(emoji_frame, dummy, None)
            print("Synthetic demo completed headlessly.")
            return
        else:
            print(f"Error: Could not open video source: {source}")
            return

    # If the user requested MJPEG streaming, run a streaming loop that
    # optionally processes frames with MediaPipe and serves them over HTTP.
    if args.mjpeg_port:
        print(f"Starting MJPEG stream on port {args.mjpeg_port} for source {source}")
        start_mjpeg_stream(cap, args.mjpeg_port, processed='auto')
        return

    if not args.show_gui or not can_show_gui():
        print("No GUI display enabled. Running headless processing for 30 frames. Use --show-gui on a GUI machine or --mjpeg-port PORT to view in a browser.")
        processed_frames = 0
        # Prefer Task API if available
        if HAS_MEDIAPIPE_TASKS:
            lm = create_task_landmarker_from_asset('face_landmarker_v2.task')
            if lm is not None:
                while cap.isOpened() and processed_frames < 30:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _ = detect_with_task_landmarker(lm, frame)
                    processed_frames += 1
                try:
                    lm.close()
                except Exception:
                    pass
                cap.release()
                print(f"Headless processing completed using tasks API ({processed_frames} frames).")
                return
        # Else try legacy solutions API
        if mp_face_mesh is not None:
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
                while cap.isOpened() and processed_frames < 30:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    processed_frames += 1
            cap.release()
            print(f"Headless processing completed ({processed_frames} frames).")
            return

        # Fall back to simple OpenCV detection
        while cap.isOpened() and processed_frames < 30:
            ret, frame = cap.read()
            if not ret:
                break
            faces = detect_faces_opencv(frame)
            processed_frames += 1
        cap.release()
        print(f"Headless processing completed with OpenCV fallback ({processed_frames} frames).")
        return

    # Set up MediaPipe Face Mesh
    # max_num_faces=1 for single face tracking, refine_landmarks for more detail
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Flip the frame horizontally for a "selfie-view" display
            frame = cv2.flip(frame, 1)
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and find faces
            results = face_mesh.process(rgb_frame)

            # Create a black background for the "emoji" face
            emoji_frame = np.zeros_like(frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the face mesh on the original frame
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())

                    # Prepare points for our simplified emoji drawing
                    h, w, _ = frame.shape
                    face_2d_points = []
                    for i, lm in enumerate(face_landmarks.landmark):
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_2d_points.append((x, y))

                    # --- Apply simplified emoji features to emoji_frame ---
                    # We draw directly onto our blank canvas to simulate an "emoji" face
                    draw_emoji_eyes(emoji_frame, face_landmarks, face_2d_points)
                    draw_emoji_mouth(emoji_frame, face_landmarks, face_2d_points)

                    # --- Get head pose and simulate different views ---
                    pitch, yaw, roll = get_head_pose(face_landmarks, w, h)

            # Display the original frame with face mesh
            cv2.imshow('Original Camera Feed with Face Mesh', frame)
            # Display the "emoji" face (our simplified merge)
            cv2.imshow('Emoji Merged Face (Simplified)', emoji_frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()