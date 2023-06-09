import os 
import json


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_label_txt", type=str, default=None)
parser.add_argument("--ann_path", type=str, default=None)

args = parser.parse_args()

label_txt = args.output_label_txt
ann_path = args.ann_path


d = {}
for fn in os.listdir(ann_path):
    name = fn[:-5]
        
    with open(os.path.join(ann_path, fn), "r") as f:
        data = json.load(f)
    f.close()

    page_count = data["metadata"]["page_count"]

    for i in range(page_count):
        label = []
        image_name = name + "_%d.jpg" % i

        field_extractions = data["field_extractions"]
        for field_extraction in field_extractions:
            page_idx = field_extraction["page"]
            if page_idx == i:
                point = field_extraction["bbox"]
                text = field_extraction["text"]
                fieldtype = field_extraction["fieldtype"]
                line_item_id = "null"
                result = {"transcription": text, "points": point, "fieldtype": fieldtype, "line_item_id": line_item_id}
                label.append(result)
        
        
        d[image_name] = label

print(len(d))

with open(label_txt, "w") as out_file:
    json.dump(d, out_file)


with open(label_txt, "r+") as file:
    content = json.load(file)
print(len(content))







