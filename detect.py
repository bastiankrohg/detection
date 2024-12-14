# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    default_model_dir = './all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--url', type=str, help='IP @ & url of camera stream', default = '192.168.0.169:8554/cam')
    parser.add_argument('--ip', type=str, help='IP @ of camera', default = '192.168.0.169')
    parser.add_argument('--port', type=str, help='Port # of camera stream', default = '8554')
    parser.add_argument('--path', type=str, help='Path to add after ip+port for camera stream', default = '/cam')
    parser.add_argument('--droidcam', action='store_true', help='If enabled, it tries to use the ip @ from the --ip arg, to connect to an alternative camera stream, from port 4747')
    parser.add_argument('--headless', action='store_false', help='If --headless is used, the script will not try to show the output on screen')

    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    phone = None
    port = None

    if args.droidcam:
        # Use phone stream
        if args.port!="8554":
            port = args.port
        else: 
            port = "4747"
        ip = args.ip
        phone = "http://" + ip + ":" + port + "/video" 

    #cap = cv2.VideoCapture(args.camera_idx)
    rtsp_url = "rtsp://rover2.local:8554/cam"
    rover_url = "rtsp://192.168.0.169:8554/cam"
    iphone = "http://192.168.0.134:4747/video"
    #rtsp_url = "rtsp://rover.local:8554/cam"
    
    #cap=cv2.VideoCapture(rtsp_url)
    #cap=cv2.VideoCapture(iphone)
    if phone != None:
        cap=cv2.VideoCapture(phone)
    else: 
        cap=cv2.VideoCapture(rover_url)
    

    #output_rtsp_url="rtsp://coral-tpu-2.local:8555/cv"
    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fps = int(cap.get(cv2.CAP_PROP_FPS))
    #output_stream = cv2.VideoWriter(output_rtsp_url,  cv2.CAP_FFMPEG,  0,  fps,  (width,  height))


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        print(objs)
        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)
        
        #forward stream test
        #output_stream.write(cv2_im)
        
        if args.headless:
            cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    
    # output_stream.release()

    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()
