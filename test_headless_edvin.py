import time
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def run_headless(frames=50):
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        face_count = 0
        for i in range(frames):
            # generate a simple synthetic frame (moving white circle)
            h, w = 480, 640
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cx = int(w/2 + 100 * np.sin(i / 6.0))
            cy = int(h/2 + 30 * np.cos(i / 8.0))
            cv2.circle(frame, (cx, cy), 60, (255, 255, 255), -1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                face_count += len(results.multi_face_landmarks)
                print(f"Frame {i}: faces detected: {len(results.multi_face_landmarks)}")
            else:
                print(f"Frame {i}: no faces")
            time.sleep(0.01)

    print(f"Headless test finished. Total faces seen across frames: {face_count}")

if __name__ == '__main__':
    run_headless(frames=50)
