#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import print_function
import os
from time import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
from time import clock 
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
# import common
import tensorrt_r8p0.samples.python.common as common
# from downloader import getFilePath

TRT_LOGGER = trt.Logger()
final_dim = [5, 10]
input_dim = [180, 320]
anchor_size = [(input_dim[0] / final_dim[0]), (input_dim[1] / final_dim[1])]
# arr = np.arange(labels.shape[0])
# np.random.shuffle(arr)

def DisplayImage(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # fig, ax = plt.subplots(1, figsize=(6, 8))
    # image = np.transpose(image.copy(), (1, 2, 0))
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.show()

def DisplayLabel(img, bboxs):
    # image = np.transpose(image.copy(), (1, 2, 0))
    # fig, ax = plt.subplots(1, figsize=(6, 8))
    # image = img
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1)
    edgecolor = [1, 0, 0]
    if len(bboxs) == 1:
        bbox = bboxs[0]
        ax.add_patch(patches.Rectangle((bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2), bbox[2], bbox[3], linewidth=1, edgecolor=edgecolor, facecolor='none'))
    elif len(bboxs) > 1:
        for bbox in bboxs:
            ax.add_patch(patches.Rectangle((bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2), bbox[2], bbox[3], linewidth=1, edgecolor=edgecolor, facecolor='none'))
    ax.imshow(image)
    plt.show()

def label_to_box_xyxy(result, threshold = 0.9):
    validation_result = []
    result_prob = []
    final_dim = [5, 10]
    for ind_row in range(final_dim[0]):
        for ind_col in range(final_dim[1]):
            grid_info = grid_cell(ind_col, ind_row)
            validation_result_cell = []
            if result[0, ind_row, ind_col] >= threshold:
                c_x = grid_info[0] + anchor_size[1]/2 + result[1, ind_row, ind_col]
                c_y = grid_info[1] + anchor_size[0]/2 + result[2, ind_row, ind_col]
                w = result[3, ind_row, ind_col] * input_dim[1]
                h = result[4, ind_row, ind_col] * input_dim[0]
                x1, y1, x2, y2 = bbox_convert(c_x, c_y, w, h)
                x1 = np.clip(x1, 0, input_dim[1])
                x2 = np.clip(x2, 0, input_dim[1])
                y1 = np.clip(y1, 0, input_dim[0])
                y2 = np.clip(y2, 0, input_dim[0])
                validation_result_cell.append(x1)
                validation_result_cell.append(y1)
                validation_result_cell.append(x2)
                validation_result_cell.append(y2)
                result_prob.append(result[0, ind_row, ind_col])
                validation_result.append(validation_result_cell)
    validation_result = np.array(validation_result)
    result_prob = np.array(result_prob)
    return validation_result, result_prob


def voting_suppression(result_box, iou_threshold = 0.5):
    votes = np.zeros(result_box.shape[0])
    for ind, box in enumerate(result_box):
        for box_validation in result_box:
            if IoU(box_validation, box) > iou_threshold:
                votes[ind] += 1
    return (-votes).argsort()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print('Completed parsing of ONNX file')
            # if builder.platform_has_fast_fp16:
            #     print('FP16')

            #     trt.BuilderFlag.FP16

            print('Buexitilding an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

# convert feature map coord to image coord
def grid_cell(cell_indx, cell_indy):
    stride_0 = anchor_size[1]
    stride_1 = anchor_size[0]
    return np.array([cell_indx * stride_0, cell_indy * stride_1, cell_indx * stride_0 + stride_0, cell_indy * stride_1 + stride_1])

# convert from [c_x, c_y, w, h] to [x_l, y_l, x_r, y_r]
def bbox_convert(c_x, c_y, w, h):
    return [c_x - w/2, c_y - h/2, c_x + w/2, c_y + h/2]

# convert from [x_l, y_l, x_r, x_r] to [c_x, c_y, w, h]
def bbox_convert_r(x_l, y_l, x_r, y_r):
    return [x_l/2 + x_r/2, y_l/2 + y_r/2, x_r - x_l, y_r - y_l]

# calculating IoU
def IoU(a, b):
    # referring to IoU algorithm in slides
    inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter_ab = inter_w * inter_h
    area_a = (a[3] - a[1]) * (a[2] - a[0])
    area_b = (b[3] - b[1]) * (b[2] - b[0])
    union_ab = area_a + area_b - inter_ab
    return inter_ab / union_ab

def assign_label(label):
    label_gt = np.zeros((5, final_dim[0], final_dim[1]))
    IoU_threshold = 0.01
    IoU_max = 0
    IoU_max_ind = [0, 0]

    for ind_row in range(final_dim[0]):
        for ind_col in range(final_dim[1]):
            label_assign = 0
            grid_info = grid_cell(ind_col, ind_row)
            label_bbox = bbox_convert(label[0], label[1], label[2], label[3])
            IoU_value = IoU(label_bbox, grid_info)
            if IoU_value > IoU_threshold:
                label_assign = 1
            if IoU_value > IoU_max:
                IoU_max = IoU_value
                IoU_max_ind[0] = ind_row
                IoU_max_ind[1] = ind_col

            # construct the gt vector
            if label_assign == 1:
                label_gt[0, ind_row, ind_col] = 1
                label_gt[1, ind_row, ind_col] = label[0] - (grid_info[0] + anchor_size[1]/2)
                label_gt[2, ind_row, ind_col] = label[1] - (grid_info[1] + anchor_size[0]/2)
                label_gt[3, ind_row, ind_col] = label[2] / float(input_dim[1])
                label_gt[4, ind_row, ind_col] = label[3] / float(input_dim[0])
    
    grid_info = grid_cell(IoU_max_ind[0], IoU_max_ind[1])
    label_gt[0, IoU_max_ind[0], IoU_max_ind[1]] = 1
    label_gt[1, IoU_max_ind[0], IoU_max_ind[1]] = label[0] - (grid_info[0] + anchor_size[1]/2)
    label_gt[2, IoU_max_ind[0], IoU_max_ind[1]] = label[1] - (grid_info[1] + anchor_size[0]/2)
    label_gt[3, IoU_max_ind[0], IoU_max_ind[1]] = label[2] / float(input_dim[1])
    label_gt[4, IoU_max_ind[0], IoU_max_ind[1]] = label[3] / float(input_dim[0])
    return label_gt

def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    # dataset_folder = 'f110_dataset_20220209/'
    # path = dataset_folder + "labels.npy" 
    # labels = np.load(path)
    # print(len(labels))
    final_dim = [5, 10]
    input_dim = [180, 320]
    # anchor_size = [(input_dim[0] / final_dim[0]), (input_dim[1] / final_dim[1])]
    # arr = np.arange(labels.shape[0])
    # np.random.shuffle(arr)

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path ='f1tenth_yolov3.onnx'
    engine_file_path = "yolov3_FP32.trt"
    # Download a dog image and save it to the following file path:
    input_image_path ='images/test_car_x60cm'
    
   
  
    
    image_raw = cv2.imread(input_image_path) 
    image = cv2.resize(image_raw, (input_dim[1], input_dim[0]))
    img_in = np.transpose(image, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)

   
     # np.transpose(image,(2,0,1))
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (180,320)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    # preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    # image_raw, image = preprocessor.process(input_image_path)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size

    # Output shapes expected by the post-processor
   
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        start = clock()
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = img_in#.reshape(1,3,180,320)
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        output_f = trt_outputs[0].reshape(1,5,5,10)
        print(output_f.shape)
        bboxs, result_prob = label_to_box_xyxy(output_f[0],0.4)
        vote_rank = voting_suppression(bboxs, 0.6)
        bbox = bboxs[vote_rank[0]]
        [c_x, c_y, w, h] = bbox_convert_r(bbox[0], bbox[1], bbox[2], bbox[3])
        bboxs_2 = np.array([[c_x, c_y, w, h]])
        end = clock()
        print(end-start)
        DisplayLabel(image_raw, bboxs_2)

        

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    # trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]


    # postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
    #                       "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
    #                                        (59, 119), (116, 90), (156, 198), (373, 326)],
    #                       "obj_threshold": 0.6,                                               # Threshold for object coverage, float value between 0 and 1
    #                       "nms_threshold": 0.5,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
    #                       "yolo_input_resolution": input_resolution_yolov3_HW}

    # postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    # boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
    # # Draw the bounding boxes onto the original input image and save it as a PNG file
    # obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
    # output_image_path = 'dog_bboxes.png'
    # obj_detected_img.save(output_image_path, 'PNG')
    # print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))

if __name__ == '__main__':
    main()