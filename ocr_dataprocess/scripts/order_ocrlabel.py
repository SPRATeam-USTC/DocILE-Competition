import os 
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ocr_label", type=str, default=None)
parser.add_argument("--ocr_label_order", type=str, default=None)
args = parser.parse_args()

ocr_label = args.ocr_label
ocr_label_order = args.ocr_label_order


with open(ocr_label, "r") as f:       
    ocr_datas = json.load(f)
    
d = {}
for image_name, ocr_data_list in ocr_datas.items():
    label = []
    points = []
    transcription = []
    confidence = []
    for ocr_data in ocr_data_list:
        points.append(ocr_data["points"])
        transcription.append(ocr_data["transcription"])
        confidence.append(ocr_data["confidence"])

    sorted_points = sorted(points, key=lambda x: (x[1], x[0])) 
    sorted_transcription = [transcription[points.index(p)] for p in sorted_points]
    sorted_confidence = [confidence[points.index(p)] for p in sorted_points]

    for x,y,z in zip(sorted_points, sorted_transcription, sorted_confidence):
        result = {"confidence": "%0.16f"%float(z), "transcription": y, "points": x}
        label.append(result)
    
    d[image_name] = label

with open(os.path.join(ocr_label_order, "train_ordered_ocrlabel_result_without_bigbox.json"), "w") as out_file:
    json.dump(d, out_file)
