import os
import json
import cv2
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default=None)
parser.add_argument("--ocr_origin_path", type=str, default=None)
parser.add_argument("--ocr_result_path", type=str, default=None)

args = parser.parse_args()

img_path = args.img_path
ocr_origin_path = args.ocr_origin_path
ocr_result_path = args.ocr_result_path



with open(ocr_origin_path,"r") as f:
    ocr_origin = json.load(f)

ocr_result = {}
threshold = 0.3
def compute_max_cover(bbox, points, w, h):
    list = [w, h, w, h]
   
    px1, py1, px2, py2 = [a*b for a,b in zip(bbox, list)]
    gx1, gy1, gx2, gy2 = [c*d for c,d in zip(points, list)]
    
    parea = (px2 - px1) * (py2 - py1)
    garea = (gx2 - gx1) * (gy2 - gy1)

    x1 = max(px1, gx1)
    y1 = max(py1, gy1)
    x2 = min(px2, gx2)
    y2 = min(py2, gy2)

    w = max(0, (x2 - x1))
    h = max(0, (y2 -y1))
    area = w * h
    iou = area / min(garea, parea)
    return iou

for image_name, info_ocr_origin in ocr_origin.items():
    label = []
    info_ocr_origin_tmp = copy.deepcopy(info_ocr_origin)
    img = cv2.imread(os.path.join(img_path, image_name))
    h,w,c = img.shape
    for box in info_ocr_origin:
        count = 0
        box_pos = box["points"]
        box_text = box["transcription"]
        box_score = box["confidence"]
        for box_tmp in info_ocr_origin_tmp:
            if compute_max_cover(box_pos, box_tmp["points"], w, h) > threshold:
                count += 1
        if count < 3:
            result = {"confidence": "%0.16f"%float(box_score), "transcription": box_text, "points": box_pos}
            label.append(result)
    ocr_result[image_name] = label

print(len(ocr_result))
with open(ocr_result_path, "w") as out_file:
    json.dump(ocr_result, out_file)

