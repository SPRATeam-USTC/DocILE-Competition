import math
import os
import tqdm
import torch
import copy
import numpy as np
from .tokenizer import mean_pooling, english_bert, english_tokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--info_origin_result', type=str, default=None)
parser.add_argument('--output_npy_path', type=str, default=None)
parser.add_argument('--output_embed_dir', type=str, default=None)

args = parser.parse_args()

info_origin_result = list(np.load(args.info_origin_result))
output_npy_path = args.output_npy_path
output_embed_dir = args.output_embed_dir
threshold = 500
sliding_windows = 70

if not os.path.exists(output_embed_dir):
    os.makedirs(output_embed_dir)

info_result = []
for info_origin in tqdm.tqdm(info_origin_result):
    num_boxes = len(info_origin["polys"])
    
    if 0 < num_boxes <= threshold:
        if len(info_origin["instance_label"]) == 0:
            continue
        if len(info_origin["line_item_label"]) > 0:
            if sum([len(x) for x in info_origin["line_item_label"]]):
                info = copy.deepcopy(info_origin)
                info_result.append(info)
            else:
                continue
        else:
            continue
        
        
    elif num_boxes > threshold:
        info = copy.deepcopy(info_origin)
        flag = [True for _ in range(len(info["instance_label"]))]
        item_flag = [True for _ in range(len(info["line_item_label"]))]
        num_group = math.ceil(num_boxes/threshold)
        sub_group = num_boxes // num_group
        for j in range(num_group):
            start = sub_group*j
            if j == num_group-1:
                sub_group = num_boxes - start - sliding_windows
            name = info["image_name"].split("/")[-1]
            
            image_name = info['image_name']
            polys = [info['polys'][i] for i in range(start, start+sub_group+sliding_windows)]
            contents = [info['contents'][i] for i in range(start, start+sub_group+sliding_windows)]
            labels = [info['labels'][i] for i in range(start, start+sub_group+sliding_windows)]
            
            instance_label_origin = info["instance_label"]
            instance_class_origin = info["instance_class"]
            line_item_label_origin = info["line_item_label"]
            instance_label_list = []
            instance_class_list = []
            line_item_label = []
            for k,instance in enumerate(instance_label_origin):
                if min(instance) >= start and max(instance) < start+sub_group+sliding_windows:
                    instance_label_list.append([(x - start) for x in instance])
                    instance_class_list.append(instance_class_origin[k])
                    flag[k] = False

            for k,line_item in enumerate(line_item_label_origin):
                if len([box for ins in line_item for box in ins]) > 0:
                    if min([box for ins in line_item for box in ins]) >= start and max([box for ins in line_item for box in ins]) < start+sub_group+sliding_windows:
                        tmp = []
                        for ins in line_item :
                            tmp.append([(box - start) for box in ins])
                        line_item_label.append(tmp)
                        item_flag[k] = False
                else:
                    line_item_label.append(line_item)
                    item_flag[k] = False

            encoded_input = english_tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
            encoded_input = encoded_input.to(english_bert.device)
            with torch.no_grad():
                model_output = english_bert(**encoded_input)
            
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = sentence_embeddings.detach().cpu()
            
            sentences_embed_path = os.path.join(output_embed_dir, name[:-4] + f'_{j}.pt')
            torch.save(sentence_embeddings, sentences_embed_path)

            sub_info = dict(image_name=image_name, polys=polys, contents=contents, labels=labels, sentences_embed_path=sentences_embed_path, \
            instance_label=instance_label_list, instance_class=instance_class_list, line_item_label=line_item_label)
            
            if len(sub_info["instance_label"]) == 0:
                continue
            if len(sub_info["line_item_label"]) > 0:
                if sum([len(x) for x in sub_info["line_item_label"]]):
                    info = copy.deepcopy(sub_info)
                    info_result.append(info)
                else:
                    continue
            else:
                continue

        if sum(flag) != 0 or sum(item_flag) != 0:
            raise ValueError("The two numbers must be equal.")

    else:
        continue

np_info_result = np.array(info_result)
np.save(output_npy_path, np_info_result)












