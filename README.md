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
python mathew.py --list-cameras
```

Open the webcam and start tracking:

```bash
python mathew.py --camera 0
```

If your USB webcam is not at index `0`, use the detected index from `--list-cameras` or a device path like `/dev/video0`.
