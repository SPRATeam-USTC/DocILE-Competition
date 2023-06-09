# goal: extract info
import os
import sys
import cv2
import json
import tqdm
import shutil
import subprocess
import numpy as np
import fitz
from PIL import Image
from libs.utils import draw_ocr_box_txt 

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_pdf_path', type=str)
    parser.add_argument('--src_ocr_path', type=str)
    parser.add_argument('--pdf_list', type=str)
    parser.add_argument('--tgt_image_path', type=str)
    parser.add_argument('--output_info_dir', type=str)
    parser.add_argument('--visualize_dir', type=str, default=None)
    parser.add_argument('--use_mp', type=bool, default=False)
    parser.add_argument('--st_idx', type=int, default=None)
    parser.add_argument('--ed_idx', type=int, default=None)
    parser.add_argument('--min_required_lines', type=int, default=5)
    parser.add_argument('--min_required_ave_confidence', type=float, default=0.85)
    parser.add_argument('--min_required_word_confidence', type=float, default=0.5)
    parser.add_argument('--min_required_chars', type=float, default=50)
    args = parser.parse_args()

    if not os.path.exists(args.tgt_image_path):
        os.makedirs(args.tgt_image_path, exist_ok=True)
    if not os.path.exists(args.output_info_dir):
        os.makedirs(args.output_info_dir, exist_ok=True)
    if args.visualize_dir != None and not os.path.exists(args.visualize_dir):
        os.makedirs(args.visualize_dir, exist_ok=True)

    return args


def single_process(args):
    image_size = (1024, 1024) # w*h
    pdf_list_basename = os.path.splitext(os.path.basename(args.pdf_list))[0]

    suffix = pdf_list_basename
    pdf_list = json.load(open(args.pdf_list))
    if args.st_idx != None and args.ed_idx != None:
        suffix = "{}_{}_{}".format(pdf_list_basename, args.st_idx, args.ed_idx)
        pdf_list = pdf_list[args.st_idx:args.ed_idx]
    cacher_path = os.path.join(args.output_info_dir, 'infos_%d.npy' % args.process_id)
    cacher = list()

    all_pdfs, all_pages, saved_pages = 0,0,0
    for pdf_base_name in tqdm.tqdm(pdf_list):
        try:
            # Open the PDF file
            pdf_path = os.path.join(args.src_pdf_path, pdf_base_name+".pdf")
            if not os.path.exists(pdf_path): continue
            doc = fitz.open(pdf_path)
            # Load json format annotation
            ocr_path = os.path.join(args.src_ocr_path, pdf_base_name+".json")
            if not os.path.exists(ocr_path): continue
            ocr_info = json.load(open(ocr_path, 'r'))
            all_pdfs += 1
            
            # Loop through json annotation
            for page_idx, page in enumerate(ocr_info['pages']):
                all_pages += 1
                line_bboxes = []
                line_contents = []
                line_confidences = []
                for block in page['blocks']:
                    for line in block['lines']:
                        box_geo = line['geometry']
                        w_min, w_max = int(box_geo[0][0]*image_size[0]), int(box_geo[1][0]*image_size[0])
                        h_min, h_max = int(box_geo[0][1]*image_size[1]), int(box_geo[1][1]*image_size[1])
                        words, words_confidences = [], []
                        for x in line['words']:
                            if x["confidence"] > args.min_required_word_confidence:
                                words.append(x['value'])
                                words_confidences.append(x['confidence'])
                        if len(words):
                            line_bboxes.append([[w_min, h_min], [w_max, h_min], [w_max, h_max], [w_min, h_max]])
                            line_contents.append(" ".join(words))
                            line_confidences.append(np.mean(words_confidences))

                
                # Don't save those low quality pages
                if len(line_bboxes) < args.min_required_lines \
                    or np.mean(line_confidences) < args.min_required_ave_confidence \
                        or np.sum([len(x) for x in line_contents]) < args.min_required_chars:
                    continue
                
                # Convert the page to a PIL Image object
                pix = doc[page_idx].get_pixmap()
                page_image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

                # Resize the image to the desired size
                page_image = page_image.resize(image_size)

                # Save the image to a file
                image_path = os.path.join(args.tgt_image_path, '{}_{}.png'.format(pdf_base_name, page_idx))
                page_image.save(image_path, format='PNG')
                cacher.append(dict(image_name=image_path, polys=line_bboxes, contents=line_contents))
                saved_pages += 1

                if args.visualize_dir is not None:
                    vis_result = draw_ocr_box_txt(Image.open(image_path), np.array(line_bboxes), line_contents)
                    cv2.imwrite(os.path.join(args.visualize_dir, '{}_{}_line.png'.format(pdf_base_name, page_idx)), vis_result)

            doc.close()


        except RuntimeError as E:
            print('error processing %s due to %s.' %(pdf_base_name, str(E)))
            continue

    np.save(cacher_path, cacher)
    print("Processed {} PDFs with {} pages, saved {} valid pages, valid ratio {}".format(all_pdfs, all_pages, saved_pages, saved_pages/all_pages))

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


if __name__ == "__main__":
    main()