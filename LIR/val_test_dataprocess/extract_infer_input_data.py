import os
import tqdm
import glob
import torch
from xml.dom.minidom import parse
import json
from .tokenizer import mean_pooling, english_bert, english_tokenizer

threshold = 0.3
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--ocr_dir', type=str, default=None)
    parser.add_argument('--output_json_path', type=str, default=None)
    parser.add_argument('--output_embed_dir', type=str, default=None)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=int(1e8))
    args = parser.parse_args()
    
    

    if not os.path.exists(args.output_embed_dir):
        os.makedirs(args.output_embed_dir)

    return args


def compute_iou(bbox, points, w, h):
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

def single_process(args):
    image_paths = glob.glob(os.path.join(args.image_dir, "*.jpg"))[args.start_index:args.end_index]
    
    output_json_path = args.output_json_path
    ocr_json = args.ocr_dir

    with open(ocr_json, "r") as f:       
        ocr_datas = json.load(f)
    result = {}
    for image_path in tqdm.tqdm(image_paths):
        image_name = os.path.basename(image_path)
        polys = []
        contents = []
        
        ocr_data_list = ocr_datas[image_name]
        if len(ocr_data_list) == 0:
            continue
        
        for i,ocr_data in enumerate(ocr_data_list):
            ocr_box = ocr_data["points"]
            ocr_text = ocr_data["transcription"]
            
            polys.append(ocr_box)
            contents.append(ocr_text)
                
        """ extract sentence embedding """
        encoded_input = english_tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(english_bert.device)
        with torch.no_grad():
            model_output = english_bert(**encoded_input)
        
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = sentence_embeddings.detach().cpu()
        output_embed_path = os.path.abspath(os.path.join(args.output_embed_dir, image_name[:-4] + '.pt'))
        torch.save(sentence_embeddings, output_embed_path) 

        info = dict(image_name=os.path.abspath(image_path), polys=polys, contents=contents, sentences_embed_path=output_embed_path)
        result[image_name] = info
    
    with open(output_json_path, "w") as f:
        json.dump(result, f)


def main():
    args = parse_args()
    single_process(args)


if __name__ == "__main__":
    main()


