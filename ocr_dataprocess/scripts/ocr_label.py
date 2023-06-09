import os 
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_label_txt", type=str, default=None)
parser.add_argument("--ocr_path", type=str, default=None)
parser.add_argument("--geometry", type=str, default=None)

args = parser.parse_args()

label_txt = args.output_label_txt
ocr_path = args.ocr_path
geometry = args.geometry


d = {}
for fn in os.listdir(ocr_path):
    name = fn[:-5]
        
    with open(os.path.join(ocr_path, fn), "r") as f:
        data = json.load(f)
    f.close()
    
    label = []
    
    image_name = name + ".jpg"
    page = data["pages"][0]

    blocks = page["blocks"]

    for block in blocks:
        lines = block["lines"]

        for line in lines:
            words = line["words"]
            for word in words:
                score = word["confidence"]
                text = word["value"]
                pos = word[geometry]
                # pos = word["snapped_geometry"]

                x1 = pos[0][0] 
                y1 = pos[0][1]
                x2 = pos[1][0] 
                y2 = pos[1][1]
                point = [x1, y1, x2, y2]

                result = {"confidence": "%0.16f"%score, "transcription": text, "points": point}

                label.append(result)
    d[image_name] = label
print(len(d))
with open(label_txt, "w") as out_file:
    json.dump(d, out_file)

