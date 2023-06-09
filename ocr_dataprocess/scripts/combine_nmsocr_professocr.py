import os
import json
import cv2

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default=None)
parser.add_argument("--nms_ocr_path", type=str, default=None)
parser.add_argument("--profess_snap_ocr_path", type=str, default=None)
parser.add_argument("--ocr_result_path", type=str, default=None)

args = parser.parse_args()

img_path = args.img_path
nms_ocr_path = args.nms_ocr_path
profess_snap_ocr_path = args.profess_snap_ocr_path
ocr_result_path = args.ocr_result_path



with open(nms_ocr_path,"r") as f:
    nms_ocr = json.load(f)

with open(profess_snap_ocr_path, "r") as g:
    profess_snap_ocr = json.load(g)

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

for image_name, info_snap_ocr in profess_snap_ocr.items():
    label = info_snap_ocr
    info_nms_ocr = nms_ocr[image_name]
    img = cv2.imread(os.path.join(img_path, image_name))
    h,w,c = img.shape
    for nms_box in info_nms_ocr:
        flag = 1
        nms_pos = nms_box["points"]
        nms_text = nms_box["transcription"]
        nms_score = nms_box["confidence"]
        for snap_box in info_snap_ocr:
            if compute_max_cover(nms_pos, snap_box["points"], w, h) > threshold:
                flag = 0
                break
        if flag:
            result = {"confidence": "%0.16f"%float(nms_score), "transcription": nms_text, "points": nms_pos}
            label.append(result)
    ocr_result[image_name] = label

print(len(ocr_result))
with open(ocr_result_path, "w") as out_file:
    json.dump(ocr_result, out_file)

