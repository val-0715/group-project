import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Pi 5 Optimization: model_complexity=1 works great on the Pi 5 CPU
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Start your USB Webcam at 1080p
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Finger Tracker Started. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Mirror the frame for natural movement
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Scale down for fast MediaPipe tracking processing (960x540 is perfect for Pi 5)
    small_frame = cv2.resize(frame, (960, 540))
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Process the hand tracking
    rgb_small.flags.writeable = False
    results = hands.process(rgb_small)
    rgb_small.flags.writeable = True

    # If a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton lines and joint dots onto your main 1080p frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- FINGER DETECTION LOGIC ---
            # Joint IDs for the tips and the knuckles right below them
            tips = [8, 12, 16, 20]          # Index, Middle, Ring, Pinky Tips
            knuckles = [6, 10, 14, 18]      # Respective PIP joints
            finger_names = ["Index", "Middle", "Ring", "Pinky"]
            
            active_fingers = []
            
            # Check the 4 main fingers
            for t, k, name in zip(tips, knuckles, finger_names):
                # MediaPipe Y-axis goes from 0 (top) to 1 (bottom).
                # If tip Y is less than knuckle Y, the finger is raised.
                if hand_landmarks.landmark[t].y < hand_landmarks.landmark[k].y:
                    active_fingers.append(name)
            
            # Special check for the Thumb (uses X-axis movement instead of Y-axis)
            # Compare Thumb Tip (4) to Thumb IP Joint (3)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            
            # Adjust thumb logic depending on which hand is facing the camera
            if thumb_tip.x > thumb_ip.x:
                active_fingers.append("Thumb")
                
            # Print the active fingers to your terminal console in real-time
            print(f"Fingers Detected Up: {', '.join(active_fingers) if active_fingers else 'None'}")
            
            # Put text directly onto the video screen showing the count
            cv2.putText(frame, f"Fingers Up: {len(active_fingers)}", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the video feed
    cv2.imshow('Pi 5 Finger Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()