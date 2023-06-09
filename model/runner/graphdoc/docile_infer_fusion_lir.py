import argparse
import dataclasses
import json
import os
import cv2
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import tqdm
from transformers import AutoConfig, AutoTokenizer
import sys
sys.path.append("/path/to/DocILE_submit/model")
from layoutlmft.models.graphdoc.modeling_confusion import InferLirMergeFusion

def show_summary(args: argparse.Namespace, filename: str):
    """Helper function showing the summary of surgery experiment instance given by runtime
    arguments

    Parameters
    ----------
    args : argparse.Namespace
        input arguments
    """ """"""
    # Helper function showing the summary of surgery experiment instance given by runtime
    # arguments
    # """
    print("-" * 50)
    print(f"{filename}")
    print("-" * 10)
    [print(f"{k.upper()}: {v}") for k, v in vars(args).items()]
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--classify_pred", type=Path, default=None)
    parser.add_argument("--logits_path", type=Path, default=None)
    args = parser.parse_args()

    print(f"{datetime.now()} Started.")

    show_summary(args, __file__)

    os.makedirs(args.output_dir, exist_ok=True)

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    config = AutoConfig.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.sentence_model)

    model = InferLirMergeFusion.from_pretrained(
        args.checkpoint, config=config
    ).to(device)

    model.eval()

    logits = np.load(args.logits_path)
    
    with open(args.classify_pred, "r") as f:
        classify_preds = json.load(f)

    input_data = json.load(open(args.input_json, "r"))
    start = 0
    end = 0
    for image_name,image_infos in tqdm.tqdm(input_data.items()):
        if image_name == "a9626563e0494c7ebb5ee7f2_1.jpg":
            continue
        box_num = len(image_infos["polys"])
        end += box_num
        logit = logits[start:end]
        classify_pred = classify_preds["classify_preds"][start:end]
        start = end

        target_h, target_w = 512, 512
        image = cv2.imread(image_infos["image_name"])
        image = cv2.resize(image, dsize=(target_w, target_h))
        image = torch.from_numpy(image.transpose(2,0,1).astype(np.float32)).unsqueeze(0).to(device)

       
        input_bboxes = torch.tensor(image_infos["polys"])
        input_bboxes[:, [0,2]] *= target_w
        input_bboxes[:, [1,3]] *= target_h
        input_bboxes = input_bboxes.round()
        global_image_tensor = torch.tensor([0, 0, target_w, target_h])
        bbox = torch.cat([global_image_tensor.unsqueeze(0), input_bboxes], dim=0).type(torch.int64).unsqueeze(0).to(device)

        attention_mask = torch.ones((1,bbox.shape[1])).to(device)
        attention_mask[0,0] = 0

        emb_path = image_infos['sentences_embed_path']
        foreground_bbox_num = input_bboxes.shape[0]
        sentence_embeddings = torch.load(emb_path)[:foreground_bbox_num]
        cls_embed = torch.zeros_like(sentence_embeddings[0])
        sentence_embeddings = torch.cat([cls_embed[None, :], sentence_embeddings], dim=0).unsqueeze(0).to(device)

        model(image=image, bbox=bbox, attention_mask=attention_mask, \
            inputs_embeds=sentence_embeddings, image_infos=image_infos, output_path=args.output_dir, logit=logit, classify_pred=classify_pred)






    