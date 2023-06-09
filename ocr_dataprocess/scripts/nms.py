import cv2
import numpy as np
import json
import os
import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default=None)
parser.add_argument("--ocrlabel_nms", type=str, default=None)
parser.add_argument("--label_125", type=str, default=None)
parser.add_argument("--label_15", type=str, default=None)
parser.add_argument("--label_175", type=str, default=None)


args = parser.parse_args()

img_path = args.img_path
ocrlabel_nms = args.ocrlabel_nms
label_125 = args.label_125
label_15 = args.label_15
label_175 = args.label_175



threshold = 0.3

d = {}

def nms(bounding_boxes, confidence_score, text, threshold, w1, h1):
    if len(bounding_boxes) == 0:
        return [], [], []
    bboxes = np.array(bounding_boxes)
    score = np.array(confidence_score)
    text = np.array(text)

    x1 = bboxes[:, 0] * w1
    y1 = bboxes[:, 1] * h1
    x2 = bboxes[:, 2] * w1
    y2 = bboxes[:, 3] * h1
    areas =(x2 - x1 + 1) * (y2 - y1 + 1)

    order = np.argsort(score)

    picked_boxes = [] 
    picked_score = [] 
    picked_text = []
    while order.size > 0:
        index = order[-1]
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_text.append(text[index])

        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h

        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_text


with open(label_125, "r", encoding="utf-8") as f4:
    ocr_data_125 = json.load(f4)
with open(label_15, "r", encoding="utf-8") as f5:
    ocr_data_15 = json.load(f5)
with open(label_175, "r", encoding="utf-8") as f6:
    ocr_data_175 = json.load(f6)


for image_name, dis_125 in tqdm.tqdm(ocr_data_125.items()):
    bounding_boxes = []
    confidence_score = []
    text = []
    label = []
    img = cv2.imread(os.path.join(img_path, image_name))
    h,w,c = img.shape
    
    dis_15 = ocr_data_15[image_name]
    dis_175 = ocr_data_175[image_name]
    dis = dis_125 + dis_15 + dis_175


    for item in dis:
        bounding_boxes.append(item["points"])
        confidence_score.append(item["confidence"])
        text.append(item["transcription"])
    
    boxes, confidences, transes = nms(bounding_boxes, confidence_score, text, threshold, w, h)

    for box, confidence, trans in zip(boxes, confidences, transes):
        
        result = {"confidence": confidence, "transcription": trans, "points": box}

        label.append(result)
    
    d[image_name] = label

print(len(d))
with open(ocrlabel_nms, "w") as file:
    json.dump(d, file)
file.close()

        
