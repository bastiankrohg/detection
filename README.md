# Object Detection Module - Documentation

This module provides an object detection demo using OpenCV and Google Coral’s Edge TPU. The script captures video frames, runs inference on the frames using a pre-trained TensorFlow Lite model, and optionally displays the results with bounding boxes.

## Prerequisites

Before running the script, ensure you have the following:

Hardware
- Google Coral Dev Board or Coral USB Accelerator
- Camera (compatible with RTSP or local video capture)

Software
- Python 3.7+
- Required Python packages:
- OpenCV (cv2)
- Coral Python libraries (pycoral)
- Pre-trained TensorFlow Lite models optimized for the Edge TPU
- Label files corresponding to the models


Installation
1.	Clone the Repository:
```bash
git clone https://github.com/bastiankrohg/rover-coral.git
cd rover-coral/object-detection
```

2.	Install Dependencies:
Use a virtual environment for isolation:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3.	Download Models and Labels:
Place the .tflite models and label files in a directory called all_models within the project folder.

## Running the Script

Basic Command

Run the object detection script with a pre-trained model:
```bash
python detect.py 
```

Optional Arguments
- --top_k: Number of top results to display (default: 3).
- --camera_idx: Camera index for local video capture (default: 0).
- --threshold: Confidence threshold for detections (default: 0.1).
- **--url:** URL of the RTSP stream (default: 192.168.0.169:8554/cam).
- **--ip, --port, --path:** Customize the camera stream address.
- **--droidcam:** Use this arg to set up for phone-based camera stream. NB! Uses IP @ from **--ip**. 
- **--headless:** Set to display frames during inference.

Key Features

RTSP Stream Support

Supports video feeds from RTSP streams, including Raspberry Pi camera modules using MediaMTX.

Real-Time Inference
- Leverages Coral Edge TPU for low-latency inference.
- Displays bounding boxes and labels on frames with high confidence.

Customizable Detection
- Works with any TFLite model compatible with the Edge TPU.
- Dynamically switches between local and remote camera sources.


## Example Usage

Custom RTSP Stream
```bash
python detect.py --url rtsp://192.168.0.169:8554/cam
```

Notes
- Ensure the Coral Edge TPU is properly configured and connected.
- Camera and network configurations may need adjustments based on your setup.

For additional help or to report issues, please contact the maintainers.
