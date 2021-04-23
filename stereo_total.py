# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Example using TF Lite to classify objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from multiprocessing import Process, Queue
import io
import time
import numpy as np
import cv2
import os
import re

from PIL import Image
from tflite_runtime.interpreter import Interpreter

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    px0 = 0
    px1 = 0
    py0 = 0
    py1 = 0
    for obj in objs:
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        if label.split(' ')[1] == 'person':
            continue
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        px0 = x0
        px1 = x1
        py0 = y0
        py1 = y1


        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im, (px1 + px0) / 2, (py1 + py0) / 2
    
def load_labels(path):
    with open(path, 'r') as f:
      return {i: line.strip()[2:] for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results 
    if output_details['dtype'] == np.uint8:
      scale, zero_point = output_details['quantization']
      output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def cam(cid, result, classify_inference_size, classify_interpreter, classify_labels):
    cap = cv2.VideoCapture(cid)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_name = 'frame' + str(cid)
    prev_time = 0
    while True:
        ret, frame = cap.read()  # Read 결과와 frame

        frame = cv2.flip(frame, 1)
        # print(type(frame))
        result.put(frame.copy())
        
        cv2_im = frame

        cur_time = time.time()
        sec = cur_time-prev_time
        prev_time = cur_time
        fps = str(round(1/sec, 1))

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        classify_img = cv2.resize(cv2_im_rgb, classify_inference_size)
        results = classify_image(classify_interpreter, classify_img)
        fps = fps + 'fps ' + classify_labels[results[0][0]] + ' ' + str(round(results[0][1] * 100, 1)) + '%'
        #cv2.putText(cv2_im, fps, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0))

        if ret:
            cv2.putText(frame, fps, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow(frame_name, frame)  # 컬러 화면 출력
            if cv2.waitKey(1) == 27:
                # result.put('STOP')
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    classify_model = './classify_model/model.tflite'
    classify_labels = './classify_model/labels.txt'
    
    detect_model = './detect_model/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
    detect_labels = './detect_model/coco_labels.txt'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=2,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='detecter score threshold')
    args = parser.parse_args()

    print('Classify loading {} with {} labels.'.format(classify_model, classify_labels))
    classify_labels = load_labels(classify_labels)
    classify_interpreter = Interpreter(classify_model)
    classify_interpreter.allocate_tensors()
    _, height, width, _ = classify_interpreter.get_input_details()[0]['shape']
    classify_inference_size = (width, height)

    print('Detect loading {} with {} labels.'.format(detect_model, detect_labels))
    detect_interpreter = make_interpreter(detect_model)
    detect_interpreter.allocate_tensors()
    detect_labels = read_label_file(detect_labels)
    detect_inference_size = input_size(detect_interpreter)
    
    result0 = Queue()
    result1 = Queue()
    th1 = Process(target=cam, args=(0, result0, classify_inference_size, classify_interpreter, classify_labels))
    th2 = Process(target=cam, args=(2, result1, classify_inference_size, classify_interpreter, classify_labels))

    th1.start()
    th2.start()
    
#    cap = cv2.VideoCapture(args.camera_idx)
#    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prev_time = 0
    while True:
        #ret, frame = cap.read()
        frame0 = result0.get()
        frame1 = result1.get()
        cur_time = time.time()
        sec = cur_time-prev_time
        prev_time = cur_time
        fps = str(round(1/sec, 1))
        
#        frame0 = cv2.flip(frame0, 1)
#        frame1 = cv2.flip(frame1, 1)
        
        #if not ret:
        #    break
        cv2_im0 = frame0
        cv2_im1 = frame1

        cv2_im_rgb0 = cv2.cvtColor(cv2_im0, cv2.COLOR_BGR2RGB)
        cv2_im_rgb1 = cv2.cvtColor(cv2_im1, cv2.COLOR_BGR2RGB)
        
#        classify_img0 = cv2.resize(cv2_im_rgb0, classify_inference_size)
#        results0 = classify_image(classify_interpreter, classify_img0)
#        # print(labels[results[0][0]])
#        fps0 = fps + 'fps ' + classify_labels[results0[0][0]] + ' ' + str(round(results0[0][1] * 100, 1)) + '%'
#        cv2.putText(cv2_im0, fps0, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0))

#        classify_img1 = cv2.resize(cv2_im_rgb1, classify_inference_size)
#        results1 = classify_image(classify_interpreter, classify_img1)
#        # print(labels[results[0][0]])
#        fps1 = fps + 'fps ' + classify_labels[results1[0][0]] + ' ' + str(round(results1[0][1] * 100, 1)) + '%'
#        cv2.putText(cv2_im1, fps1, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0))
        
        detect_img0 = cv2.resize(cv2_im_rgb0, detect_inference_size)
        run_inference(detect_interpreter, detect_img0.tobytes())
        objs0 = get_objects(detect_interpreter, args.threshold)[:args.top_k]
        cv2_im0, pointx0, pointy0 = append_objs_to_img(cv2_im0, detect_inference_size, objs0, detect_labels)
        
        detect_img1 = cv2.resize(cv2_im_rgb1, detect_inference_size)
        run_inference(detect_interpreter, detect_img1.tobytes())
        objs1 = get_objects(detect_interpreter, args.threshold)[:args.top_k]
        cv2_im1, pointx1, pointy1 = append_objs_to_img(cv2_im1, detect_inference_size, objs1, detect_labels)

        if pointx0 > 0 and pointx1 > 0:
            print('distance:', round(3600 / abs(pointx0 - pointx1), 1), 'cm')
        
        cv2.imshow('inf0', cv2_im0)
        cv2.imshow('inf1', cv2_im1)
        if cv2.waitKey(1) & 0xFF == 27:
            break

#    cap.release()
    cv2.destroyAllWindows()

    
    th1.join()
    th2.join()


if __name__ == '__main__':
  main()
