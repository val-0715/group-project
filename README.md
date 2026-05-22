# group-project

## Body Tracking Demo

This repo contains a starter script for local webcam-based full-body pose tracking using OpenCV and MediaPipe.

### Install

```bash
pip install -r requirements.txt
```

### Run

List cameras:

```bash
python main.py --list-cameras
```

Open the webcam and start tracking:

```bash
python main.py --camera 0
```

Headless / CI-friendly usage (no webcam required):

```bash
# run a single-frame headless test and save annotated image
python main.py --headless --output demo_output.png
```

Use a video file instead of a webcam:

```bash
python main.py --video /path/to/sample.mp4
```

If your USB webcam is not at index `0`, use the detected index from `--list-cameras` or a device path like `/dev/video0`.
