# group-project

This repo contains two demo scripts:

- `mathew.py` — body/pose tracking demo using MediaPipe Pose.
- `edvin.py` — face/emoji overlay demo with flexible camera/backends and MJPEG streaming.

### Install

```bash
pip install -r requirements.txt
```

### `mathew.py` quick run

List cameras:

```bash
python mathew.py --list-cameras
```

Open the webcam and start tracking:

```bash
python mathew.py --camera 0
```

### `edvin.py` quick run

List cameras (indices 0..15):

```bash
python edvin.py --list-cameras
```

Open your camera (index 0):

```bash
python edvin.py --camera 0
```

If your terminal/container has no display, use `--show-gui` on a GUI machine or use MJPEG streaming:

```bash
python edvin.py --camera 0 --show-gui
python edvin.py --camera 0 --mjpeg-port 8080
```

Stream camera to browser as MJPEG (no GUI required):

```bash
python edvin.py --camera 0 --mjpeg-port 8080
# then open http://localhost:8080/
```

### GUI

A simple graphical launcher is available in `edvin_gui.py`:

```bash
python edvin_gui.py
```

The GUI lets you select a camera or input file, choose a backend, enable preview windows, and start/stop the demo.

Force capture backend (macOS/Windows/Linux):

```bash
python edvin.py --camera 0 --backend avfoundation   # macOS
python edvin.py --camera 0 --backend dshow         # Windows
python edvin.py --camera 0 --backend v4l2          # Linux
```

Notes:
- If `mediapipe.solutions` is not available in your installation, `edvin.py` falls back to OpenCV face detection for camera mode and still supports preview or MJPEG streaming.
- Use `--mjpeg-port` to view the camera remotely (forward the port if running in a container).

If you'd like a more detailed helper or platform-specific instructions, tell me which OS and environment (local mac, remote container, codespace, etc.).
