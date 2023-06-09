# goal: extract info
import os
import sys
import cv2
import glob
import tqdm
import torch
import subprocess
import numpy as np
from PIL import Image
from libs.tokenizer import mean_pooling, tokenizer, sentence_bert
from libs.utils import draw_ocr_box_txt 

def str2bool(v):
    return v.lower() in ("true", "t", "1")

def rec_to_poly(boxes):
    x1 = boxes[:, 0::2].min(-1)
    y1 = boxes[:, 1::2].min(-1)
    x2 = boxes[:, 0::2].max(-1)
    y2 = boxes[:, 1::2].max(-1)
    poly = np.concatenate([x1[:,None], y1[:, None], x2[:, None], y1[:, None], \
        x2[:, None], y2[:, None], x1[:, None], y2[:, None]],axis=-1)
    return poly.reshape(-1,4,2)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ocr_path', type=str, default=None)
    parser.add_argument('--output_info_dir', type=str, default=None)
    parser.add_argument('--output_embed_dir', type=str, default=None)
    parser.add_argument('--visualize_dir', type=str, default=None)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=int(1e8))
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument('--process_id', type=int, default=0)
    parser.add_argument('--total_process_num', type=int, default=1)
    args = parser.parse_args()
    
    if args.visualize_dir is not None:
        if not os.path.exists(args.visualize_dir):
            os.makedirs(args.visualize_dir)
    
    if not os.path.exists(args.output_info_dir):
        os.makedirs(args.output_info_dir)

    if not os.path.exists(args.output_embed_dir):
        os.makedirs(args.output_embed_dir)

    return args


def single_process(args):
    ocr_loader = np.load(args.ocr_path)
    valid_ids = list(range(len(ocr_loader)))[args.start_index:args.end_index]
    valid_ids = valid_ids[args.process_id::args.total_process_num]
    cacher_path = os.path.join(args.output_info_dir, 'infos_%d.npy' % args.process_id)
    cacher = list()
    for idx in tqdm.tqdm(valid_ids):
        try:
            info = ocr_loader.get_record(idx)
            image_path = info['image_name']
            image_name = os.path.basename(image_path)
            assert os.path.exists(image_path)

            contents = info['contents']
            polys = np.array(info['polys']).astype('int64')

            if len(polys) < 1: # remove empty document
                continue

            bboxes = list()
            for poly in polys:
                bboxes.append([int(poly[:, 0].min()), int(poly[:, 1].min()), \
                    int(poly[:, 0].max()), int(poly[:, 1].max())])

            if args.visualize_dir is not None:
                vis_result = draw_ocr_box_txt(Image.open(image_path), rec_to_poly(np.array(bboxes)), contents)
                cv2.imwrite(os.path.join(args.visualize_dir, image_name+'.png'), vis_result)
            
            encoded_input = tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
            encoded_input = encoded_input.to(sentence_bert.device)

            with torch.no_grad():
                model_output = sentence_bert(**encoded_input)
            
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = sentence_embeddings.detach().cpu().numpy()
            output_embed_path = os.path.abspath(os.path.join(args.output_embed_dir, image_name + '.npy'))
            np.save(output_embed_path, sentence_embeddings)

            assert len(sentence_embeddings) == len(bboxes)
            cacher.append(dict(sentences_embed=output_embed_path, bboxes=bboxes, image_name=image_path))

        except RuntimeError as E:
            print('error processing the index=%d of ocr-loader due to %s.' %(idx, str(E)))
            continue
    np.save(cacher_path, cacher)


def main():
    args = parse_args()
    if not args.use_mp:
        single_process(args)
    else:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()

    # append mask token to output embed dir
    mask_content = tokenizer.decode([tokenizer.bos_token_id, tokenizer.mask_token_id, tokenizer.sep_token_id])
    encoded_input = tokenizer(mask_content, padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(sentence_bert.device)

    with torch.no_grad():
        model_output = sentence_bert(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = sentence_embeddings.detach().cpu().numpy()
    output_embed_path = os.path.join(args.output_embed_dir, 'mask_embedding.npy')
    np.save(output_embed_path, sentence_embeddings)


if __name__ == "__main__":
    main()