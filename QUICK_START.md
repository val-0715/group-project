# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Running the Scripts

### Option 1: Face Overlay Demo
```bash
python edvin.py --camera 0 --show-gui
```

### Option 2: Body Pose Tracking
```bash
python mathew.py --camera 0
```

### Option 3: GUI Launcher
```bash
python edvin_gui.py
```

### Option 4: Remote Streaming (MJPEG)
```bash
python edvin.py --camera 0 --mjpeg-port 8080
# Then open: http://localhost:8080/
```

### Option 5: Headless Processing
```bash
python edvin.py          # Face detection
python mathew.py         # Pose tracking
python test_headless_edvin.py  # Test
```

## System Requirements

- Python 3.10+
- OpenCV with camera support
- MediaPipe
- Webcam or video input device

## Troubleshooting

### Camera Not Found
- Check camera index: `python edvin.py --list-cameras`
- Use correct index: `python edvin.py --camera <INDEX>`

### No Display Available
- Use `--mjpeg-port PORT` for browser viewing
- Or run in headless mode (no display needed)

### Performance Issues
- Reduce resolution
- Lower detection confidence thresholds
- Use lighter model (edvin.py has multiple options)

---
See TESTING_REPORT.md for detailed test results
