# goal: extract info
import os
import cv2
import tqdm
import glob
import torch
import numpy as np
import json
from .tokenizer import mean_pooling, english_bert, english_tokenizer


threshold = 0.3
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--ocr_dir', type=str, default=None)
    parser.add_argument('--ann_txt', type=str, default=None)
    parser.add_argument('--output_npy_path', type=str, default=None)
    parser.add_argument('--output_embed_dir', type=str, default=None)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=int(1e8))
    args = parser.parse_args()
    

    if not os.path.exists(args.output_embed_dir):
        os.makedirs(args.output_embed_dir)

    return args


def compute_iou(bbox, points, w, h):
    wh = [w, h, w, h]
   
    px1, py1, px2, py2 = [a*b for a,b in zip(bbox, wh)]
    gx1, gy1, gx2, gy2 = [c*d for c,d in zip(points, wh)]
    
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

def single_process(args):
    image_paths = glob.glob(os.path.join(args.image_dir, "*.jpg"))[args.start_index:args.end_index]
    
    output_npy_path = args.output_npy_path
    ocr_txt = args.ocr_dir
    ann_txt = args.ann_txt

    with open(ocr_txt, "r") as f:       
        ocr_datas = json.load(f)
    with open(ann_txt, "r") as f1:
        ann_datas = json.load(f1)


    info_result =  []
   
    for image_path in tqdm.tqdm(image_paths):
        try:
            image_name = os.path.basename(image_path)
            img = cv2.imread(image_path)
            h,w,c = img.shape
            polys = []
            contents = []
            labels = []
            in_instances = []
            ocr_data_list = ocr_datas[image_name]
            if len(ocr_data_list) == 0:
                continue
            ann_data_list = ann_datas[image_name]
                    
            for i,ocr_data in enumerate(ocr_data_list):
                tmp_label = []
                in_instance = []
                ocr_box = ocr_data["points"]
                ocr_text = ocr_data["transcription"]
                if len(ann_data_list)==0:
                    tmp_label.append("other")
                    in_instance.append("null")
                    polys.append(ocr_box)
                    contents.append(ocr_text)
                    labels.append(tmp_label)
                    in_instances.append(in_instance)
                else:                   
                    for j,ann_data in enumerate(ann_data_list):
                        ann_box = ann_data["points"]
                        ann_label = ann_data["fieldtype"]

                        if compute_iou(ann_box, ocr_box, w, h) > threshold and ann_label not in tmp_label:
                            tmp_label.append(ann_label)
                            in_instance.append(j)

                    if len(tmp_label) == 0:
                        tmp_label.append("other")
                        in_instance.append("null")
                        polys.append(ocr_box)
                        contents.append(ocr_text)
                        labels.append(tmp_label)
                        in_instances.append(in_instance)

                    else:
                        polys.append(ocr_box)
                        contents.append(ocr_text)
                        labels.append(tmp_label)
                        in_instances.append(in_instance)
            
            instances_member = list(set(x for instance in in_instances for x in instance if x!= "null"))

            instance_label_list_tmp = [[] for _ in range(len(ann_data_list))]
            for i, instance in enumerate(instances_member):
                for j, in_instance in enumerate(in_instances):
                    if instance in in_instance:
                        instance_label_list_tmp[instance].append(j)
            
            instance_class_list_tmp = [ann_data["fieldtype"] for ann_data in ann_data_list]
            instance_label_list = []
            instance_class_list = []
            for i, (instance_label, instance_class) in enumerate(zip(instance_label_list_tmp, instance_class_list_tmp)):
                if len(instance_label) == 0:
                    continue
                if sorted(instance_label) not in instance_label_list:
                    instance_label_list.append(sorted(instance_label))
                    instance_class_list.append(instance_class)
                if sorted(instance_label) in instance_label_list and instance_class_list[instance_label_list.index(sorted(instance_label))] != instance_class:
                    instance_label_list.append(sorted(instance_label))
                    instance_class_list.append(instance_class)
          
            """ extract sentence embedding """
            encoded_input = english_tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
            encoded_input = encoded_input.to(english_bert.device)
            with torch.no_grad():
                model_output = english_bert(**encoded_input)
            
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = sentence_embeddings.detach().cpu()
            output_embed_path = os.path.abspath(os.path.join(args.output_embed_dir, image_name[:-4] + '.pt'))
            torch.save(sentence_embeddings, output_embed_path) # temp = torch.load(output_embed_path)

            info = dict(image_name=os.path.abspath(image_path), polys=polys, contents=contents, labels=labels, sentences_embed_path=output_embed_path, instance_label=instance_label_list, instance_class=instance_class_list)

            info_result.append(info)
           
        except RuntimeError as E:
            print('error processing %s due to %s.' %(image_path, str(E)))
            continue

    np_info_result = np.array(info_result)
    np.save(output_npy_path, np_info_result)



def main():
    args = parse_args()
    single_process(args)


if __name__ == "__main__":
    main()


